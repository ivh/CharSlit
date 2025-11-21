#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstring>

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
    int maxiter
) {
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
    if (slitcurve.shape(0) != ncols || slitcurve.shape(1) != 3) {
        throw std::runtime_error("slitcurve must have shape (ncols, 3)");
    }

    // Calculate ny (size of slit function array)
    int ny = osample * (nrows + 1) + 1;

    if (slitdeltas.shape(0) != ny) {
        throw std::runtime_error("slitdeltas must have length ny = osample * (nrows + 1) + 1");
    }

    // Allocate output arrays
    size_t sP_shape[1] = {static_cast<size_t>(ncols)};
    size_t sL_shape[1] = {static_cast<size_t>(ny)};
    size_t model_shape[2] = {static_cast<size_t>(nrows), static_cast<size_t>(ncols)};
    size_t unc_shape[1] = {static_cast<size_t>(ncols)};
    size_t info_shape[1] = {5};

    double* sP = new double[ncols];
    double* sL = new double[ny];
    double* model = new double[nrows * ncols];
    double* unc = new double[ncols];
    double* info = new double[5];

    // Initialize sP and sL with ones (starting guess)
    for (int i = 0; i < ncols; i++) {
        sP[i] = 1.0;
    }
    for (int i = 0; i < ny; i++) {
        sL[i] = 1.0 / osample;
    }

    // Create a copy of mask since slitdec modifies it
    unsigned char* mask_copy = new unsigned char[nrows * ncols];
    std::memcpy(mask_copy, mask.data(), nrows * ncols * sizeof(unsigned char));

    // Create a copy of ycen since slitdec modifies it
    double* ycen_copy = new double[ncols];
    std::memcpy(ycen_copy, ycen.data(), ncols * sizeof(double));

    // Call the C function
    int result = slitdec(
        ncols,
        nrows,
        const_cast<double*>(im.data()),
        const_cast<double*>(pix_unc.data()),
        mask_copy,
        ycen_copy,
        const_cast<double*>(slitcurve.data()),
        const_cast<double*>(slitdeltas.data()),
        osample,
        lambda_sP,
        lambda_sL,
        maxiter,
        sP,
        sL,
        model,
        unc,
        info
    );

    // Create nanobind arrays that own the data
    nb::capsule sP_owner(sP, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule sL_owner(sL, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule model_owner(model, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule unc_owner(unc, [](void *p) noexcept { delete[] (double *) p; });
    nb::capsule info_owner(info, [](void *p) noexcept { delete[] (double *) p; });
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

    // Create result dictionary
    nb::dict result_dict;
    result_dict["spectrum"] = sP_array;
    result_dict["slitfunction"] = sL_array;
    result_dict["model"] = model_array;
    result_dict["uncertainty"] = unc_array;
    result_dict["info"] = info_array;
    result_dict["mask"] = mask_array;
    result_dict["return_code"] = result;

    return result_dict;
}

NB_MODULE(slitchar, m) {
    m.def("slitdec", &slitdec_wrapper,
          nb::arg("im"),
          nb::arg("pix_unc"),
          nb::arg("mask"),
          nb::arg("ycen"),
          nb::arg("slitcurve"),
          nb::arg("slitdeltas"),
          nb::arg("osample") = 6,
          nb::arg("lambda_sP") = 0.0,
          nb::arg("lambda_sL") = 0.1,
          nb::arg("maxiter") = 20,
          "Slit decomposition with slit characterization\n\n"
          "Parameters\n"
          "----------\n"
          "im : ndarray (nrows, ncols)\n"
          "    Main input image to be decomposed\n"
          "pix_unc : ndarray (nrows, ncols)\n"
          "    Pixel error map\n"
          "mask : ndarray (nrows, ncols), uint8\n"
          "    Pixel mask (will be modified by the algorithm)\n"
          "ycen : ndarray (ncols,)\n"
          "    Order centre line offset from pixel row boundary\n"
          "slitcurve : ndarray (ncols, 3)\n"
          "    Polynomial coefficients for slit curvature\n"
          "slitdeltas : ndarray (ny,)\n"
          "    Slit deltas, where ny = osample * (nrows + 1) + 1\n"
          "osample : int, optional\n"
          "    Subpixel oversampling factor (default: 6)\n"
          "lambda_sP : float, optional\n"
          "    Smoothing parameter for spectrum (default: 0.0)\n"
          "lambda_sL : float, optional\n"
          "    Smoothing parameter for slit function (default: 0.1)\n"
          "maxiter : int, optional\n"
          "    Maximum number of iterations (default: 20)\n\n"
          "Returns\n"
          "-------\n"
          "dict with keys:\n"
          "    spectrum : ndarray (ncols,)\n"
          "        Extracted spectrum\n"
          "    slitfunction : ndarray (ny,)\n"
          "        Slit illumination function\n"
          "    model : ndarray (nrows, ncols)\n"
          "        Model reconstructed from spectrum and slit function\n"
          "    uncertainty : ndarray (ncols,)\n"
          "        Spectrum uncertainties\n"
          "    info : ndarray (5,)\n"
          "        Status information [success, cost, status, iter, delta_x]\n"
          "    mask : ndarray (nrows, ncols), uint8\n"
          "        Updated pixel mask after outlier rejection\n"
          "    return_code : int\n"
          "        Return code from C function (0 on success)\n"
    );
}
