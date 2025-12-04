"""Tests for the slitdec nanobind wrapper."""

import numpy as np
import pytest
import charslit


class TestBasicFunctionality:
    """Test basic functionality of the slitdec wrapper."""

    def test_import_module(self):
        """Test that the module can be imported."""
        assert hasattr(charslit, "slitdec")

    def test_function_callable(self):
        """Test that slitdec is callable."""
        assert callable(charslit.slitdec)

    @pytest.mark.save_output
    def test_basic_execution(self, simple_image_data, save_test_data):
        """Test that slitdec runs without errors on valid input."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )
        assert isinstance(result, dict)
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"],
                      spectrum=result["spectrum"],
                      slitfunction=result["slitfunction"],
                      uncertainty=result["uncertainty"])

    @pytest.mark.save_output
    def test_with_custom_parameters(self, simple_image_data, save_test_data):
        """Test slitdec with custom parameters."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
            lambda_sP=0.01,
            lambda_sL=0.05,
            maxiter=10,
        )
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"],
                      spectrum=result["spectrum"],
                      slitfunction=result["slitfunction"],
                      uncertainty=result["uncertainty"])


class TestOutputStructure:
    """Test the structure and types of outputs."""

    def test_output_keys(self, simple_image_data, save_test_data):
        """Test that all expected keys are present in output."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        expected_keys = {
            "spectrum",
            "slitfunction",
            "model",
            "uncertainty",
            "info",
            "mask",
            "return_code",
        }
        assert set(result.keys()) == expected_keys
        save_test_data(data["im"], result["model"])

    def test_output_types(self, simple_image_data, save_test_data):
        """Test that output arrays have correct types."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        assert isinstance(result["spectrum"], np.ndarray)
        assert isinstance(result["slitfunction"], np.ndarray)
        assert isinstance(result["model"], np.ndarray)
        assert isinstance(result["uncertainty"], np.ndarray)
        assert isinstance(result["info"], np.ndarray)
        assert isinstance(result["mask"], np.ndarray)
        assert isinstance(result["return_code"], int)
        save_test_data(data["im"], result["model"])

    def test_output_shapes(self, simple_image_data, save_test_data):
        """Test that output arrays have correct shapes."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        assert result["spectrum"].shape == (data["ncols"],)
        assert result["slitfunction"].shape == (data["ny"],)
        assert result["model"].shape == (data["nrows"], data["ncols"])
        assert result["uncertainty"].shape == (data["ncols"],)
        assert result["info"].shape == (5,)
        assert result["mask"].shape == (data["nrows"], data["ncols"])
        save_test_data(data["im"], result["model"])

    def test_output_dtypes(self, simple_image_data, save_test_data):
        """Test that output arrays have correct dtypes."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        assert result["spectrum"].dtype == np.float64
        assert result["slitfunction"].dtype == np.float64
        assert result["model"].dtype == np.float64
        assert result["uncertainty"].dtype == np.float64
        assert result["info"].dtype == np.float64
        assert result["mask"].dtype == np.uint8
        save_test_data(data["im"], result["model"])


class TestInputValidation:
    """Test input validation and error handling."""

    def test_mismatched_image_shapes(self, simple_image_data):
        """Test that mismatched image shapes raise an error."""
        data = simple_image_data
        wrong_pix_unc = np.ones((10, 10))  # Wrong shape

        with pytest.raises(RuntimeError, match="pix_unc must have same shape"):
            charslit.slitdec(
                data["im"],
                wrong_pix_unc,
                data["mask"],
                data["ycen"],
                data["slitcurve"],
                data["slitdeltas"],
            )

    def test_mismatched_mask_shape(self, simple_image_data):
        """Test that mismatched mask shape raises an error."""
        data = simple_image_data
        wrong_mask = np.ones((10, 10), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="mask must have same shape"):
            charslit.slitdec(
                data["im"],
                data["pix_unc"],
                wrong_mask,
                data["ycen"],
                data["slitcurve"],
                data["slitdeltas"],
            )

    def test_wrong_ycen_length(self, simple_image_data):
        """Test that wrong ycen length raises an error."""
        data = simple_image_data
        wrong_ycen = np.ones(10)  # Wrong length

        with pytest.raises(RuntimeError, match="ycen must have length ncols"):
            charslit.slitdec(
                data["im"],
                data["pix_unc"],
                data["mask"],
                wrong_ycen,
                data["slitcurve"],
                data["slitdeltas"],
            )

    def test_wrong_slitcurve_shape(self, simple_image_data):
        """Test that wrong slitcurve shape raises an error."""
        data = simple_image_data
        wrong_slitcurve = np.zeros((10, 3))  # Wrong shape

        with pytest.raises(RuntimeError, match="slitcurve must have shape"):
            charslit.slitdec(
                data["im"],
                data["pix_unc"],
                data["mask"],
                data["ycen"],
                wrong_slitcurve,
                data["slitdeltas"],
            )

    def test_wrong_slitdeltas_length(self, simple_image_data):
        """Test that wrong slitdeltas length raises an error."""
        data = simple_image_data
        wrong_slitdeltas = np.zeros(10)  # Wrong length

        with pytest.raises(RuntimeError, match="slitdeltas must have length"):
            charslit.slitdec(
                data["im"],
                data["pix_unc"],
                data["mask"],
                data["ycen"],
                data["slitcurve"],
                wrong_slitdeltas,
            )


class TestNumericalBehavior:
    """Test numerical behavior of the algorithm."""

    def test_spectrum_normalization(self, simple_image_data, save_test_data):
        """Test that the extracted spectrum is properly normalized."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
        )

        # Spectrum should be positive and finite
        assert np.all(np.isfinite(result["spectrum"]))
        # Most values should be positive (allowing for some edge effects)
        assert np.sum(result["spectrum"] > 0) > 0.8 * len(result["spectrum"])
        save_test_data(data["im"], result["model"], spectrum=result["spectrum"])

    def test_slitfunction_normalization(self, simple_image_data, save_test_data):
        """Test that the slit function is normalized."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
        )

        # Slit function should integrate to approximately osample
        integral = np.sum(result["slitfunction"])
        assert np.isclose(integral, data["osample"], rtol=0.1)
        save_test_data(data["im"], result["model"], slitfunction=result["slitfunction"])

    def test_model_reconstruction(self, simple_image_data, save_test_data):
        """Test that the model has correct properties."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        # Model should be finite
        assert np.all(np.isfinite(result["model"]))

        # Model should roughly match input image where mask is valid
        masked_input = data["im"][data["mask"] > 0]
        masked_model = result["model"][result["mask"] > 0]
        # Correlation should be positive
        if len(masked_input) > 0 and len(masked_model) > 0:
            correlation = np.corrcoef(masked_input, masked_model)[0, 1]
            assert correlation > 0.5, "Model should correlate with input"
        save_test_data(data["im"], result["model"])

    def test_info_array_content(self, simple_image_data, save_test_data):
        """Test that info array contains meaningful values."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        info = result["info"]
        # info[0] should be success flag (1 for success)
        assert info[0] in [0, 1], "Success flag should be 0 or 1"
        # info[1] is cost (should be finite and positive)
        assert (
            np.isfinite(info[1]) and info[1] >= 0
        ), "Cost should be finite and non-negative"
        # info[3] is iteration count (should be positive)
        assert info[3] > 0, "Should have run at least one iteration"
        save_test_data(data["im"], result["model"], info=result["info"])

    @pytest.mark.save_output
    def test_mask_modification(self, simple_image_data, save_test_data):
        """Test that mask is modified to reject outliers."""
        data = simple_image_data

        # Add some outliers
        im_with_outliers = data["im"].copy()
        im_with_outliers[5, 10] = 1e6  # Large outlier
        im_with_outliers[7, 15] = -1e6  # Large negative outlier

        result = charslit.slitdec(
            im_with_outliers,
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        # Output mask should have rejected some pixels
        assert np.sum(result["mask"]) < np.sum(
            data["mask"]
        ), "Mask should reject some outlier pixels"
        save_test_data(im_with_outliers, result["model"],
                      spectrum=result["spectrum"],
                      slitfunction=result["slitfunction"],
                      uncertainty=result["uncertainty"],
                      mask_out=result["mask"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_image(self, minimal_image_data, save_test_data):
        """Test with minimal valid image size."""
        data = minimal_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
        )
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"])

    def test_with_curvature(self, curved_image_data, save_test_data):
        """Test with slit curvature."""
        data = curved_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
        )
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"], slitcurve=data["slitcurve"])

    def test_high_smoothing(self, simple_image_data, save_test_data):
        """Test with high smoothing parameters."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            lambda_sP=1.0,
            lambda_sL=1.0,
        )
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"])

    def test_low_iterations(self, simple_image_data, save_test_data):
        """Test with low iteration count."""
        data = simple_image_data
        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            maxiter=2,
        )
        # Should still complete, but might not converge
        assert result["return_code"] == 0
        # Iteration count should be close to maxiter (allowing for off-by-one)
        assert result["info"][3] <= 3  # Should be close to maxiter
        save_test_data(data["im"], result["model"])

    def test_different_osample_values(self, simple_image_data, save_test_data):
        """Test with different oversampling factors."""
        data = simple_image_data
        for osample in [3, 4, 8]:
            ny = osample * (data["nrows"] + 1) + 1
            slitdeltas = np.zeros(ny)

            result = charslit.slitdec(
                data["im"],
                data["pix_unc"],
                data["mask"],
                data["ycen"],
                data["slitcurve"],
                slitdeltas,
                osample=osample,
            )
            assert result["return_code"] == 0
            assert result["slitfunction"].shape == (ny,)
            # Only save for osample=4 to avoid too many files
            if osample == 4:
                save_test_data(data["im"], result["model"])


class TestDefaultParameters:
    """Test default parameter behavior."""

    def test_default_osample(self, simple_image_data, save_test_data):
        """Test that default osample=6 works correctly."""
        data = simple_image_data
        # Create slitdeltas for osample=6
        osample = 6
        ny = osample * (data["nrows"] + 1) + 1
        slitdeltas = np.zeros(ny)

        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            slitdeltas,
        )
        assert result["return_code"] == 0
        save_test_data(data["im"], result["model"])

    def test_explicit_vs_default_parameters(self, simple_image_data, save_test_data):
        """Test that explicit parameters match defaults."""
        data = simple_image_data

        # Use defaults
        result1 = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"].copy(),
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        # Use explicit defaults
        result2 = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"].copy(),
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=6,
            lambda_sP=0.0,
            lambda_sL=0.1,
            maxiter=20,
        )

        # Results should be identical
        np.testing.assert_array_equal(result1["spectrum"], result2["spectrum"])
        np.testing.assert_array_equal(result1["slitfunction"], result2["slitfunction"])
        save_test_data(data["im"], result1["model"])


class TestMemoryManagement:
    """Test memory management and data ownership."""

    def test_input_arrays_unchanged(self, simple_image_data, save_test_data):
        """Test that input arrays are not modified (except mask)."""
        data = simple_image_data

        # Make copies to check
        im_copy = data["im"].copy()
        pix_unc_copy = data["pix_unc"].copy()
        ycen_copy = data["ycen"].copy()
        slitcurve_copy = data["slitcurve"].copy()
        slitdeltas_copy = data["slitdeltas"].copy()

        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        # Check that inputs are unchanged
        np.testing.assert_array_equal(data["im"], im_copy)
        np.testing.assert_array_equal(data["pix_unc"], pix_unc_copy)
        np.testing.assert_array_equal(data["ycen"], ycen_copy)
        np.testing.assert_array_equal(data["slitcurve"], slitcurve_copy)
        np.testing.assert_array_equal(data["slitdeltas"], slitdeltas_copy)
        save_test_data(data["im"], result["model"])

    def test_multiple_calls(self, simple_image_data, save_test_data):
        """Test that multiple calls work correctly."""
        data = simple_image_data

        result1 = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"].copy(),
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        result2 = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"].copy(),
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
        )

        # Results should be identical
        np.testing.assert_array_equal(result1["spectrum"], result2["spectrum"])
        np.testing.assert_array_equal(result1["slitfunction"], result2["slitfunction"])
        save_test_data(data["im"], result1["model"])


class TestRealData:
    """Test slitdec on real data from FITS files."""

    @pytest.mark.save_output
    def test_real_data_files(self, real_data_files, save_test_data):
        """Test slitdec on real FITS data from data/ directory."""
        data = real_data_files

        result = charslit.slitdec(
            data["im"],
            data["pix_unc"],
            data["mask"],
            data["ycen"],
            data["slitcurve"],
            data["slitdeltas"],
            osample=data["osample"],
        )

        # Basic checks
        assert result["return_code"] == 0, f"slitdec failed on {data['filename']}"
        assert np.all(
            np.isfinite(result["spectrum"])
        ), f"Non-finite values in spectrum for {data['filename']}"
        assert np.all(
            np.isfinite(result["model"])
        ), f"Non-finite values in model for {data['filename']}"

        # Save data with filename in the extra arrays
        save_test_data(
            data["im"],
            result["model"],
            spectrum=result["spectrum"],
            slitfunction=result["slitfunction"],
            uncertainty=result["uncertainty"],
        )
