#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <cmath>
#include <cstdio>

extern "C" {
#include "extract.h"
}

namespace nb = nanobind;
using namespace nb::literals;

// Use simplified ndarray types
using ndarray_double_2d = nb::ndarray<double, nb::ndim<2>, nb::device::cpu>;
using ndarray_double_1d = nb::ndarray<double, nb::ndim<1>, nb::device::cpu>;
using ndarray_uint8_2d = nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cpu>;
using ndarray_int_2d = nb::ndarray<nb::numpy, nb::ndim<2>, nb::device::cpu>;

nb::tuple extract_wrapper(
    ndarray_double_2d im,
    ndarray_double_2d pix_unc,
    ndarray_int_2d mask,
    ndarray_double_1d ycen,
    ndarray_double_1d slitdeltas,
    int osample,
    double lambda_sP,
    double lambda_sL,
    int maxiter,
    nb::object slit_func_in = nb::none()
) {
    // Extract dimensions from image
    if (im.ndim() != 2) {
        throw std::runtime_error("Input image must be 2-dimensional");
    }

    int nrows = static_cast<int>(im.shape(0));
    int ncols = static_cast<int>(im.shape(1));

    // Calculate ny (needed for output sL array)
    int ny = osample * (nrows + 1) + 1;

    // Create output arrays using numpy arrays - allocate with new
    double *sP_data = new double[ncols];
    double *sL_data = new double[ny];
    double *model_data = new double[nrows * ncols];
    double *unc_data = new double[ncols];
    double *info_data = new double[5];
    double *img_mad_data = new double[nrows * ncols];
    uint8_t *img_mad_mask_data = new uint8_t[nrows * ncols];

    size_t shape_1d_sP[1] = {static_cast<size_t>(ncols)};
    size_t shape_1d_sL[1] = {static_cast<size_t>(ny)};
    size_t shape_2d[2] = {static_cast<size_t>(nrows), static_cast<size_t>(ncols)};
    size_t shape_info[1] = {5};

    auto sP = nb::ndarray<nb::numpy, double>(sP_data, 1, shape_1d_sP);
    auto sL = nb::ndarray<nb::numpy, double>(sL_data, 1, shape_1d_sL);
    auto model = nb::ndarray<nb::numpy, double>(model_data, 2, shape_2d);
    auto unc = nb::ndarray<nb::numpy, double>(unc_data, 1, shape_1d_sP);
    auto info = nb::ndarray<nb::numpy, double>(info_data, 1, shape_info);
    auto img_mad = nb::ndarray<nb::numpy, double>(img_mad_data, 2, shape_2d);
    auto img_mad_mask = nb::ndarray<nb::numpy, uint8_t>(img_mad_mask_data, 2, shape_2d);

    // Get buffer pointers for all arrays
    auto im_ptr = im.data();
    auto pix_unc_ptr = pix_unc.data();
    auto ycen_ptr = ycen.data();
    auto slitdeltas_ptr = slitdeltas.data();

    // Convert mask to uint8_t format that C code expects
    // Allocate a temporary buffer for the mask
    unsigned char *mask_uint8_ptr = new unsigned char[nrows * ncols];

    // Handle different input mask types - convert to uint8
    nb::dlpack::dtype mask_dtype = mask.dtype();
    if (mask_dtype.code == (uint8_t)nb::dlpack::dtype_code::Int && mask_dtype.bits == 32) {
        const int32_t* mask_int32 = static_cast<const int32_t*>(mask.data());
        for (int i = 0; i < nrows * ncols; i++) {
            mask_uint8_ptr[i] = mask_int32[i] != 0 ? 1 : 0;
        }
    } else if (mask_dtype.code == (uint8_t)nb::dlpack::dtype_code::UInt && mask_dtype.bits == 8) {
        const uint8_t* mask_u8 = static_cast<const uint8_t*>(mask.data());
        std::memcpy(mask_uint8_ptr, mask_u8, nrows * ncols);
    } else {
        delete[] mask_uint8_ptr;
        throw std::runtime_error("Unsupported mask dtype - must be int32 or uint8");
    }

    // Use the raw data pointers directly
    auto sP_ptr = sP_data;
    auto sL_ptr = sL_data;
    auto model_ptr = model_data;
    auto unc_ptr = unc_data;
    auto info_ptr = info_data;
    auto img_mad_ptr = img_mad_data;
    auto img_mad_mask_ptr = img_mad_mask_data;

    // If slit_func_in is provided, initialize sL with it
    if (!slit_func_in.is_none()) {
        auto slit_func_in_arr = nb::cast<ndarray_double_1d>(slit_func_in);
        auto slit_func_in_ptr = slit_func_in_arr.data();
        memcpy(sL_ptr, slit_func_in_ptr, ny * sizeof(double));
    }

    // Call the C extract function
    int result = extract(
        ncols, nrows,
        im_ptr, pix_unc_ptr, mask_uint8_ptr,
        ycen_ptr, slitdeltas_ptr,
        osample, lambda_sP, lambda_sL, maxiter,
        sP_ptr, sL_ptr, model_ptr, unc_ptr, info_ptr
    );

    // Calculate img_mad as absolute difference between image and model
    for (int i = 0; i < nrows * ncols; i++) {
        img_mad_ptr[i] = std::abs(im_ptr[i] - model_ptr[i]);
        // Simple example mask based on a threshold
        img_mad_mask_ptr[i] = (img_mad_ptr[i] > 0.1) ? 1 : 0;
    }

    // Clean up temporary mask buffer
    delete[] mask_uint8_ptr;

    // Return the result code and all output arrays
    return nb::make_tuple(
        result,
        sL,
        sP,
        model,
        unc,
        info,
        img_mad,
        img_mad_mask
    );
}

NB_MODULE(_charslit, m) {
    m.doc() = "CharSlit: Python wrapper for characterized slit spectral extraction";

    m.def("extract", &extract_wrapper,
        "im"_a,
        "pix_unc"_a,
        "mask"_a,
        "ycen"_a,
        "slitdeltas"_a,
        "osample"_a,
        "lambda_sP"_a,
        "lambda_sL"_a,
        "maxiter"_a,
        "slit_func_in"_a = nb::none(),
        "Extract spectrum from a 2D image using optimal extraction algorithm");
}
