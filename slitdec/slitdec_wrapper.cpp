#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <cstring>
#include <optional>
#include <string>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <vector>

extern "C" {
#include "slitdec.h"
}

namespace nb = nanobind;

nb::dict slitdec_wrapper(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> im,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> pix_unc,
    nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> mask,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> ycen,
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> slitcurve,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> slitdeltas,
    int osample,
    double lambda_sP,
    double lambda_sL,
    int maxiter,
    double kappa,
    std::optional<nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>> preset_slitfunc,
    std::string weights,
    std::optional<double> noise_floor,
    int osample_x,
    std::optional<std::vector<double>> mu_x_in,
    std::optional<std::vector<double>> mu_y,
    std::optional<int> y_lower_lim,
    std::optional<int> out_x,
    std::string uncertainty,
    int nlag
) {
    int unc_mode;
    if (uncertainty == "residual")
        unc_mode = 0;
    else if (uncertainty == "propagated")
        unc_mode = 1;
    else if (uncertainty == "scaled")
        unc_mode = 2;
    else
        throw std::runtime_error("uncertainty must be 'residual', 'propagated' or 'scaled'");
    if (nlag < 0)
        throw std::runtime_error("nlag must be >= 0");
    if (out_x.has_value() && (*out_x < 1 || osample_x % *out_x != 0))
        throw std::runtime_error("out_x must be >= 1 and divide osample_x");
    if (osample_x < 1)
        throw std::runtime_error("osample_x must be >= 1");
    // Without a penalty the fine bins are not determined (exact null modes), so
    // osample_x > 1 gets the noise-whitening penalty unless mu_x is given
    const std::vector<double> mu_x = mu_x_in.has_value() ? *mu_x_in
        : (osample_x > 1 ? std::vector<double>{0.0, 0.0, 1.64e-4} : std::vector<double>());
    if (mu_x.size() > SLITDEC_OX_KMAX || (mu_y.has_value() && mu_y->size() > SLITDEC_OX_KMAX))
        throw std::runtime_error("mu_x and mu_y take at most 15 derivative orders");
    int weight_mode;
    if (weights == "none")
        weight_mode = 0;
    else if (weights == "unc")
        weight_mode = 1;
    else if (weights == "model")
        weight_mode = 2;
    else if (weights == "flux")
        weight_mode = 3;
    else
        throw std::runtime_error("weights must be 'none', 'unc', 'model' or 'flux'");
    // Get dimensions from input image
    int nrows = im.shape(0);
    int ncols = im.shape(1);

    // Validate input dimensions
    if (pix_unc.shape(0) != nrows || pix_unc.shape(1) != ncols) {
        throw std::runtime_error("pix_unc must have same shape as im");
    }
    if (mask.shape(0) != nrows || mask.shape(1) != ncols) {
        throw std::runtime_error("mask must have same shape as im");
    }
    if (ycen.shape(0) != ncols) {
        throw std::runtime_error("ycen must have length ncols");
    }
    // slitcurve must have shape (ncols, n_coeffs) where n_coeffs <= 6
    int n_coeffs_in = slitcurve.shape(1);
    if (slitcurve.shape(0) != ncols || n_coeffs_in > 6 || n_coeffs_in < 1) {
        throw std::runtime_error("slitcurve must have shape (ncols, n) where 1 <= n <= 6");
    }

    // Calculate ny (size of slit function array)
    int ny = osample * (nrows + 1) + 1;

    // slitdeltas can be either nrows or ny length
    // If nrows, we interpolate to ny
    int slitdeltas_len = slitdeltas.shape(0);
    double* slitdeltas_interp = nullptr;
    bool allocated_interp = false;

    if (slitdeltas_len == nrows) {
        // Copy input data first
        const double* slitdeltas_data = slitdeltas.data();
        double* slitdeltas_in = new double[nrows];
        for (int i = 0; i < nrows; i++) {
            slitdeltas_in[i] = slitdeltas_data[i];
        }

        // Linearly interpolate from nrows to ny
        slitdeltas_interp = new double[ny];
        allocated_interp = true;

        for (int i = 0; i < ny; i++) {
            // Map i (0 to ny-1) to position in original array (0 to nrows-1)
            double pos = i * (nrows - 1.0) / (ny - 1.0);
            int idx_low = static_cast<int>(pos);
            int idx_high = idx_low + 1;

            if (idx_high >= nrows) {
                // Edge case: at the end
                slitdeltas_interp[i] = slitdeltas_in[nrows - 1];
            } else {
                // Linear interpolation
                double frac = pos - idx_low;
                slitdeltas_interp[i] = (1.0 - frac) * slitdeltas_in[idx_low]
                                      + frac * slitdeltas_in[idx_high];
            }
        }
        delete[] slitdeltas_in;
    } else if (slitdeltas_len == ny) {
        // Already correct length, make a copy
        slitdeltas_interp = new double[ny];
        allocated_interp = true;
        std::memcpy(slitdeltas_interp, slitdeltas.data(), ny * sizeof(double));
    } else {
        throw std::runtime_error("slitdeltas must have length nrows or ny = osample * (nrows + 1) + 1");
    }

    // Allocate output arrays
    size_t sP_shape[1] = {static_cast<size_t>(ncols)};
    size_t sL_shape[1] = {static_cast<size_t>(ny)};
    size_t model_shape[2] = {static_cast<size_t>(nrows), static_cast<size_t>(ncols)};
    size_t unc_shape[1] = {static_cast<size_t>(ncols)};
    size_t info_shape[1] = {5};
    size_t sPf_shape[1] = {static_cast<size_t>(ncols) * osample_x};
    size_t ext_shape[1] = {6};

    // Value-initialized so early failure returns (e.g. excessive curvature)
    // hand zeros to Python instead of uninitialized memory
    double* sP = new double[ncols];
    double* sL = new double[ny];
    double* model = new double[nrows * ncols]();
    double* unc = new double[ncols]();
    double* info = new double[5];
    double* sPf = new double[static_cast<size_t>(ncols) * osample_x]();
    double* info_ext = new double[6];
    double info_all[11] = {0};
    info_all[4] = -11;  // ask slitdec for info[5..10]

    // Initialize sP with ones (starting guess)
    for (int i = 0; i < ncols; i++) {
        sP[i] = 1.0;
    }

    // Seed sL. With a preset slit function we copy it into sL and tell the
    // C code to skip the sL solve (single-pass extraction). The preset may be
    // length ny (used directly) or nrows (linearly interpolated up, mirroring
    // the slitdeltas handling above). Otherwise seed the flat 1/osample guess.
    int use_preset = 0;
    if (preset_slitfunc.has_value()) {
        use_preset = 1;
        const auto& ps = preset_slitfunc.value();
        int ps_len = ps.shape(0);
        const double* ps_data = ps.data();
        if (ps_len == ny) {
            std::memcpy(sL, ps_data, ny * sizeof(double));
        } else if (ps_len == nrows) {
            for (int i = 0; i < ny; i++) {
                double pos = i * (nrows - 1.0) / (ny - 1.0);
                int idx_low = static_cast<int>(pos);
                int idx_high = idx_low + 1;
                if (idx_high >= nrows) {
                    sL[i] = ps_data[nrows - 1];
                } else {
                    double frac = pos - idx_low;
                    sL[i] = (1.0 - frac) * ps_data[idx_low] + frac * ps_data[idx_high];
                }
            }
        } else {
            // Free already-allocated buffers before throwing
            if (allocated_interp && slitdeltas_interp != nullptr) delete[] slitdeltas_interp;
            delete[] sP; delete[] sL; delete[] model; delete[] unc; delete[] info;
            delete[] sPf; delete[] info_ext;
            throw std::runtime_error("preset_slitfunc must have length nrows or ny = osample * (nrows + 1) + 1");
        }
    } else {
        for (int i = 0; i < ny; i++) {
            sL[i] = 1.0 / osample;
        }
    }

    // Create a copy of mask since slitdec modifies it
    unsigned char* mask_copy = new unsigned char[nrows * ncols];
    std::memcpy(mask_copy, mask.data(), nrows * ncols * sizeof(unsigned char));

    // Create a copy of ycen since slitdec modifies it
    double* ycen_copy = new double[ncols];
    std::memcpy(ycen_copy, ycen.data(), ncols * sizeof(double));

    // Pad slitcurve to 6 coefficients per column (C code uses stride 6)
    double* slitcurve_padded = nullptr;
    bool allocated_slitcurve = false;
    if (n_coeffs_in != 6) {
        slitcurve_padded = new double[ncols * 6];
        allocated_slitcurve = true;
        // Initialize all to zero
        std::memset(slitcurve_padded, 0, ncols * 6 * sizeof(double));
        // Copy input coefficients (up to 6)
        int n_to_copy = (n_coeffs_in < 6) ? n_coeffs_in : 6;
        for (int i = 0; i < ncols; i++) {
            for (int j = 0; j < n_to_copy; j++) {
                slitcurve_padded[i * 6 + j] = slitcurve.data()[i * n_coeffs_in + j];
            }
        }
    } else {
        slitcurve_padded = const_cast<double*>(slitcurve.data());
    }

    // Call the C function
    const int cov_x = out_x.has_value() ? *out_x : 1;
    const size_t ncov = static_cast<size_t>(ncols) * cov_x;
    double* cov = unc_mode > 0 ? new double[ncov * (nlag + 1)]() : nullptr;
    double chi2 = 0.0;
    const std::vector<double>& muy = mu_y.has_value() ? *mu_y : std::vector<double>();
    int result = slitdec(
        ncols, nrows,
        const_cast<double*>(im.data()),
        const_cast<double*>(pix_unc.data()),
        mask_copy, ycen_copy, slitcurve_padded, slitdeltas_interp,
        osample, lambda_sP, lambda_sL, maxiter, kappa,
        use_preset, weight_mode,
        noise_floor.has_value() ? *noise_floor : -1.0,
        sP, sL, model, unc, info_all,
        osample_x,
        mu_x.data(), static_cast<int>(mu_x.size()),
        muy.data(), static_cast<int>(muy.size()),
        y_lower_lim.has_value() ? *y_lower_lim : -1,
        sPf,
        unc_mode, cov_x, nlag, cov, &chi2
    );
    std::memcpy(info, info_all, 5 * sizeof(double));
    std::memcpy(info_ext, info_all + 5, 6 * sizeof(double));

    // Clean up interpolated slitdeltas
    if (allocated_interp && slitdeltas_interp != nullptr) {
        delete[] slitdeltas_interp;
    }

    // Clean up padded slitcurve
    if (allocated_slitcurve && slitcurve_padded != nullptr) {
        delete[] slitcurve_padded;
    }

    // ycen_copy is not returned to Python, unlike mask_copy
    delete[] ycen_copy;

    // Create nanobind arrays that own the data
    nb::capsule sP_owner(sP, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule sL_owner(sL, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule model_owner(model, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule unc_owner(unc, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule info_owner(info, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule sPf_owner(sPf, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule ext_owner(info_ext, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule mask_owner(mask_copy, [](void *p) noexcept { delete[] (unsigned char *) p; });

    auto sP_array = nb::ndarray<nb::numpy, double>(
        sP, 1, sP_shape, sP_owner);
    auto sL_array = nb::ndarray<nb::numpy, double>(
        sL, 1, sL_shape, sL_owner);
    auto model_array = nb::ndarray<nb::numpy, double>(
        model, 2, model_shape, model_owner);
    auto unc_array = nb::ndarray<nb::numpy, double>(
        unc, 1, unc_shape, unc_owner);
    auto info_array = nb::ndarray<nb::numpy, double>(
        info, 1, info_shape, info_owner);
    auto mask_array = nb::ndarray<nb::numpy, uint8_t>(
        mask_copy, 2, model_shape, mask_owner);

    auto sPf_array = nb::ndarray<nb::numpy, double>(
        sPf, 1, sPf_shape, sPf_owner);
    auto ext_array = nb::ndarray<nb::numpy, double>(
        info_ext, 1, ext_shape, ext_owner);

    // Create result dictionary
    nb::dict result_dict;
    result_dict["spectrum"] = sP_array;
    result_dict["slitfunction"] = sL_array;
    result_dict["model"] = model_array;
    result_dict["uncertainty"] = unc_array;
    result_dict["spectrum_fine"] = sPf_array;
    if (out_x.has_value()) {
        // Mean of osample_x / out_x adjacent fine bins: the model's density over each output bin
        const int nb = osample_x / *out_x;
        const size_t nout = static_cast<size_t>(ncols) * *out_x;
        double* sPo = new double[nout];
        for (size_t j = 0; j < nout; j++) {
            double acc = 0.0;
            for (int k = 0; k < nb; k++) acc += sPf[j * nb + k];
            sPo[j] = acc / nb;
        }
        size_t out_shape[1] = {nout};
        nb::capsule out_owner(sPo, [](void *p) noexcept { delete[] (double *) p; });
        result_dict["spectrum_out"] = nb::ndarray<nb::numpy, double>(sPo, 1, out_shape, out_owner);
    }
    if (unc_mode > 0) {
        size_t cov_shape[2] = {ncov, static_cast<size_t>(nlag + 1)};
        nb::capsule cov_owner(cov, [](void *p) noexcept { delete[] (double *) p; });
        result_dict["covariance"] = nb::ndarray<nb::numpy, double>(cov, 2, cov_shape, cov_owner);
        result_dict["chi2"] = chi2;
    }
    result_dict["info"] = info_array;
    result_dict["margin"] = static_cast<int>(info_all[4]);
    result_dict["info_ox"] = ext_array;
    result_dict["mask"] = mask_array;
    result_dict["return_code"] = result;

    return result_dict;
}

static int edge_margin_wrapper(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> slitcurve,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> slitdeltas,
    int nrows,
    int osample = 6,
    std::optional<int> y_lower_lim = std::nullopt
) {
    const int ncols = slitcurve.shape(0);
    const int nc = slitcurve.shape(1);
    const int ny = osample * (nrows + 1) + 1;
    const int nd = slitdeltas.shape(0);
    if (nrows < 1 || osample < 1)
        throw std::runtime_error("nrows and osample must be >= 1");
    if (nc > 6 || nc < 1)
        throw std::runtime_error("slitcurve must have shape (ncols, n) where 1 <= n <= 6");
    if (nd != nrows && nd != ny)
        throw std::runtime_error("slitdeltas must have length nrows or ny = osample * (nrows + 1) + 1");

    std::vector<double> curve(static_cast<size_t>(ncols) * 6, 0.0);
    for (int i = 0; i < ncols; i++)
        for (int j = 0; j < nc; j++)
            curve[i * 6 + j] = slitcurve.data()[i * nc + j];
    // Same interpolation as slitdec_wrapper, so the result equals info[4]
    std::vector<double> deltas(ny);
    for (int i = 0; i < ny; i++) {
        if (nd == ny) {
            deltas[i] = slitdeltas.data()[i];
            continue;
        }
        double pos = i * (nrows - 1.0) / (ny - 1.0);
        int lo = static_cast<int>(pos);
        if (lo + 1 >= nrows) {
            deltas[i] = slitdeltas.data()[nrows - 1];
        } else {
            double frac = pos - lo;
            deltas[i] = (1.0 - frac) * slitdeltas.data()[lo] + frac * slitdeltas.data()[lo + 1];
        }
    }
    return slitdec_margin(ncols, nrows, curve.data(), deltas.data(), osample,
                          y_lower_lim.has_value() ? *y_lower_lim : -1);
}

NB_MODULE(slitdec, m) {
    m.def("edge_margin", &edge_margin_wrapper,
          nb::arg("slitcurve"),
          nb::arg("slitdeltas"),
          nb::arg("nrows"),
          nb::arg("osample") = 6,
          nb::arg("y_lower_lim") = nb::none(),
          "Columns at either end of a swath that slitdec's solution should not be\n"
          "used for: the largest slit-image shift from slitcurve (ncols, n) and\n"
          "slitdeltas (nrows or ny), rounded up, plus 2. Equals result['margin'] of\n"
          "slitdec for the same inputs; use it to lay out overlapping swaths before\n"
          "the solve.\n");
    m.def("slitdec", &slitdec_wrapper,
          nb::arg("im"),
          nb::arg("pix_unc"),
          nb::arg("mask"),
          nb::arg("ycen"),
          nb::arg("slitcurve"),
          nb::arg("slitdeltas"),
          nb::arg("osample") = 6,
          nb::arg("lambda_sP") = 0.0,
          nb::arg("lambda_sL") = 1.0,
          nb::arg("maxiter") = 20,
          nb::arg("kappa") = 10.0,
          nb::arg("preset_slitfunc") = nb::none(),
          nb::arg("weights") = "none",
          nb::arg("noise_floor") = nb::none(),
          nb::arg("osample_x") = 1,
          nb::arg("mu_x") = nb::none(),
          nb::arg("mu_y") = nb::none(),
          nb::arg("y_lower_lim") = nb::none(),
          nb::arg("out_x") = nb::none(),
          nb::arg("uncertainty") = "residual",
          nb::arg("nlag") = 2,
          "Slit decomposition, optionally with the spectrum oversampled along x\n\n"
          "Parameters\n"
          "----------\n"
          "im : ndarray (nrows, ncols)\n"
          "    Main input image to be decomposed\n"
          "pix_unc : ndarray (nrows, ncols)\n"
          "    Pixel error map. Used for weights='unc', and to fit the variance\n"
          "    model of weights='model'; output uncertainties are estimated from\n"
          "    the data - model residuals either way.\n"
          "mask : ndarray (nrows, ncols), uint8\n"
          "    Pixel mask (will be modified by the algorithm)\n"
          "ycen : ndarray (ncols,)\n"
          "    Order centre line offset from pixel row boundary\n"
          "slitcurve : ndarray (ncols, n) where n <= 6\n"
          "    Polynomial coefficients for slit curvature (c0..c5), padded to 6\n"
          "slitdeltas : ndarray (nrows,) or (ny,)\n"
          "    Slit deltas. Can be length nrows (will be linearly interpolated to ny)\n"
          "    or length ny = osample * (nrows + 1) + 1 (used directly)\n"
          "osample : int, optional\n"
          "    Subpixel oversampling factor (default: 6)\n"
          "lambda_sP : float, optional\n"
          "    slitdec's unscaled first-difference penalty on the spectrum, applied\n"
          "    between fine bins (default: 0.0). Prefer mu_x.\n"
          "lambda_sL : float, optional\n"
          "    Smoothing parameter for slit function (default: 1.0)\n"
          "maxiter : int, optional\n"
          "    Maximum number of iterations (default: 20)\n"
          "kappa : float, optional\n"
          "    Outlier rejection threshold in sigma units (default: 10.0).\n"
          "    Pixels with |residual| > kappa * sigma are masked. Set to 0 to disable.\n"
          "preset_slitfunc : ndarray (nrows,) or (ny,), optional\n"
          "    Fixed slit function for single-pass extraction. When given, the\n"
          "    slit-function solve is skipped and only the spectrum is fitted\n"
          "    against this profile. Length nrows is interpolated to ny; the\n"
          "    preset is normalized to sum osample internally. Default: None.\n\n"
          "weights : {'none', 'unc', 'model', 'flux'}, optional\n"
          "    Pixel weights in the sL and sP solves (normalised to mean 1).\n"
          "    'none': unweighted least squares, the historic behaviour: every row\n"
          "    enters sP with weight ~sL, so its line profile counts with ~sL^2.\n"
          "    'unc': static 1/pix_unc^2. Uncertainties computed from the noisy data\n"
          "    are correlated with the noise and distort line profiles.\n"
          "    'model': first iteration unweighted, then 1/(a*max(model, 0) + b)\n"
          "    refreshed every iteration, with a, b fitted to pix_unc^2 on binned\n"
          "    medians. Noise-optimal only as far as pix_unc describes the real noise.\n"
          "    'flux': first iteration unweighted, then 1/v with v the slit-function\n"
          "    value of the pixel (floored at 1e-3 of its maximum). Rows enter sP in\n"
          "    proportion to their flux at any S/N: the bright-source LSF, independent\n"
          "    of S/N, at some noise cost for faint data. No units. Default 'none'.\n"
          "noise_floor : float, optional\n"
          "    Variance offset b in the units of im for weights='model': read\n"
          "    noise plus the shot noise of any subtracted background. Default None:\n"
          "    fitted, but not below the 5th percentile of pix_unc^2.\n"
          "osample_x : int, optional\n"
          "    Fine spectrum bins per detector column. Default 1: one bin per\n"
          "    column with the linear two-column split, the historic slitdec.\n"
          "    8 is the recommended value otherwise (docs/slitdec_ox.md): it removes\n"
          "    the line-shape error and, with the default mu_x, the excess noise\n"
          "    of the linear split on a tilted slit.\n"
          "mu_x : sequence of float, optional\n"
          "    Spectrum penalties on the 1st, 2nd, ... derivative in physical units:\n"
          "    transfer 1/(1 + sum mu_k nu^2k), nu in rad/px, independent of\n"
          "    osample, osample_x, flux and weights. Default None: no penalty for\n"
          "    osample_x = 1, (0, 0, 1.64e-4) otherwise, which whitens the extraction\n"
          "    noise for any tilt. (0, 0, 1e-6) keeps sub-pixel line shape at the\n"
          "    price of that noise. Never empty with osample_x > 1: the fine bins\n"
          "    are then not determined.\n"
          "mu_y : sequence of float, optional\n"
          "    Slit-function penalties, same units. None (default): lambda_sL.\n"
          "    mu_y=(1e-2,) is the recommended start (docs/slitdec_ox.md).\n"
          "y_lower_lim : int, optional\n"
          "    Rows below the trace row in the swath (default None: nrows // 2).\n"
          "out_x : int, optional\n"
          "    Also return the spectrum on out_x bins per column (key spectrum_out):\n"
          "    the mean of osample_x / out_x adjacent fine bins. Must divide\n"
          "    osample_x. Default None: not returned.\n"
          "uncertainty : {'residual', 'propagated', 'scaled'}, optional\n"
          "    'residual' (default): from the data - model residuals per column, as\n"
          "    slitdec; blind to the noise correlation a tilted slit introduces.\n"
          "    'propagated': the pixel variances (pix_unc^2, or the fitted a*model+b\n"
          "    with weights='model') carried through the spectrum solve, including\n"
          "    tilt, weights, mask and penalty; the slit function is taken as known.\n"
          "    Also returns covariance and chi2. 'scaled': the same times the reduced\n"
          "    chi^2 of the residuals, for pix_unc of unknown scale.\n"
          "nlag : int, optional\n"
          "    Number of lags in the returned covariance (default 2).\n\n"
          "The defaults (osample_x=1, no mu_x) are the historic slitdec, except that the\n"
          "tilt is evaluated at the y-subpixel centre (the historic code used its lower\n"
          "edge, shifting lines by c1/(2*osample)).\n\n"
          "Returns\n"
          "-------\n"
          "dict with keys:\n"
          "    spectrum : ndarray (ncols,)\n"
          "        Extracted spectrum on the detector grid (mean of the fine bins).\n"
          "        Do not use the first and last `margin` columns.\n"
          "    spectrum_fine : ndarray (ncols * osample_x,)\n"
          "        Spectrum on the fine grid\n"
          "    spectrum_out : ndarray (ncols * out_x,), only with out_x\n"
          "        Spectrum on out_x bins per column\n"
          "    slitfunction : ndarray (ny,)\n"
          "        Slit illumination function\n"
          "    model : ndarray (nrows, ncols)\n"
          "        Model reconstructed from spectrum and slit function\n"
          "    uncertainty : ndarray (ncols,)\n"
          "        Spectrum uncertainties\n"
          "    covariance : ndarray (ncols * out_x, nlag + 1), only if not 'residual'\n"
          "        covariance[j, l] = Cov(bin j, bin j + l) on the out_x grid\n"
          "        (detector grid without out_x); column 0 is the variance\n"
          "    chi2 : float, only if not 'residual'\n"
          "        Reduced chi^2 of data - model against the pixel variances used\n"
          "    info : ndarray (5,)\n"
          "        Status information [success, cost, status, iter, margin].\n"
          "        status: 1 converged, -1 out of iterations,\n"
          "        -2 curvature too large, -3 non-finite result, -4 out of memory\n"
          "    margin : int\n"
          "        Columns at either end of spectrum (and of the other per-column\n"
          "        outputs) that should not be used: the largest slit-image shift plus\n"
          "        2. They are returned as solved, not zeroed, and can be far off.\n"
          "    info_ox : ndarray (6,)\n"
          "        [dropped zeta entries, sP band half-width, damping ratio x,\n"
          "        nominal resolution x [px], damping ratio y, resolution y [px]].\n"
          "        Legacy lambda_sL: damping y = -(equivalent mu1_y).\n"
          "    mask : ndarray (nrows, ncols), uint8\n"
          "        Updated pixel mask after outlier rejection\n"
          "    return_code : int\n"
          "        Return code from C function (0 on success)\n"
    );
}
