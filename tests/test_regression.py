"""Golden-reference regression tests.

These pin the exact numerical output of slitdec so that optimization or
refactoring work can be verified not to change the extraction results.

Generate / regenerate the reference files with:

    uv run pytest tests/test_regression.py --update-golden

The synthetic inputs below are intentionally frozen (independent of the
conftest fixtures) so that edits to the general-purpose fixtures cannot
silently invalidate the golden references.
"""

from pathlib import Path

import numpy as np
import pytest

import charslit

REFERENCE_DIR = Path(__file__).parent / "reference_data"
COMPARED_KEYS = ("spectrum", "slitfunction", "model", "uncertainty")
RTOL = 1e-10


@pytest.fixture
def golden_check(request):
    update = request.config.getoption("--update-golden")

    def check(name, result):
        ref_path = REFERENCE_DIR / f"{name}.npz"
        arrays = {k: result[k] for k in COMPARED_KEYS}
        arrays["mask"] = result["mask"]

        if update:
            REFERENCE_DIR.mkdir(exist_ok=True)
            np.savez_compressed(ref_path, **arrays)
            pytest.skip(f"updated golden reference {ref_path.name}")

        if not ref_path.exists():
            pytest.fail(
                f"Missing golden reference {ref_path}. "
                "Generate it with: uv run pytest tests/test_regression.py --update-golden"
            )

        with np.load(ref_path) as ref:
            for key in COMPARED_KEYS:
                # atol scaled to the data magnitude so near-zero entries
                # don't fail on meaningless absolute noise
                scale = np.nanmax(np.abs(ref[key])) if ref[key].size else 1.0
                np.testing.assert_allclose(
                    arrays[key],
                    ref[key],
                    rtol=RTOL,
                    atol=RTOL * max(scale, 1.0),
                    err_msg=f"{name}: '{key}' deviates from golden reference",
                )
            np.testing.assert_array_equal(
                arrays["mask"],
                ref["mask"],
                err_msg=f"{name}: output mask deviates from golden reference",
            )

    return check


def _simple_case():
    """Gaussian spectrum x Gaussian slit, no curvature, seeded noise."""
    nrows, ncols, osample = 20, 50, 6
    x = np.arange(ncols)
    spectrum = np.exp(-0.5 * ((x - ncols / 2) / 5) ** 2) * 100 + 10
    ny = osample * (nrows + 1) + 1
    y = np.arange(ny) / osample - nrows / 2
    slitfunc = np.exp(-0.5 * (y / 2) ** 2)
    slitfunc /= slitfunc.sum() / osample

    im = np.zeros((nrows, ncols))
    for j in range(nrows):
        slit_contrib = slitfunc[j * osample : (j + 1) * osample + 1].sum()
        im[j, :] = spectrum * slit_contrib
    im += np.random.RandomState(42).normal(0, 1, im.shape)

    return {
        "im": im,
        "pix_unc": np.ones_like(im),
        "mask": np.ones(im.shape, dtype=np.uint8),
        "ycen": np.full(ncols, 0.5),
        "slitcurve": np.zeros((ncols, 3)),
        "slitdeltas": np.zeros(ny),
        "kwargs": {"osample": osample},
    }


def _curved_slitdeltas_case():
    """Curvature polynomial plus nonzero slitdeltas (length nrows, exercising
    the wrapper's interpolation path) and non-default smoothing."""
    nrows, ncols, osample = 15, 40, 4
    rng = np.random.RandomState(123)
    im = rng.randn(nrows, ncols) * 5 + 100

    slitcurve = np.zeros((ncols, 3))
    slitcurve[:, 1] = 0.01
    slitcurve[:, 2] = 0.001

    return {
        "im": im,
        "pix_unc": np.ones_like(im) * 2,
        "mask": np.ones(im.shape, dtype=np.uint8),
        "ycen": np.linspace(0.3, 0.7, ncols),
        "slitcurve": slitcurve,
        "slitdeltas": 0.3 * np.sin(np.arange(nrows) / 3.0),
        "kwargs": {"osample": osample, "lambda_sP": 0.01, "lambda_sL": 0.05},
    }


def _masked_case():
    """Seeded noisy image with bad pixels and a fully masked column."""
    nrows, ncols, osample = 12, 30, 6
    rng = np.random.RandomState(7)
    im = rng.randn(nrows, ncols) * 3 + 50
    im[5, 10] += 500  # outlier for kappa rejection to act on

    mask = np.ones(im.shape, dtype=np.uint8)
    mask[:, 20] = 0
    mask[3, 7] = 0

    ny = osample * (nrows + 1) + 1
    return {
        "im": im,
        "pix_unc": np.sqrt(np.abs(im) + 1.0),
        "mask": mask,
        "ycen": np.full(ncols, 0.5),
        "slitcurve": np.zeros((ncols, 3)),
        "slitdeltas": np.zeros(ny),
        "kwargs": {"osample": osample, "kappa": 5.0},
    }


SYNTHETIC_CASES = {
    "simple": _simple_case,
    "curved_slitdeltas": _curved_slitdeltas_case,
    "masked": _masked_case,
}


@pytest.mark.parametrize("case_name", sorted(SYNTHETIC_CASES))
def test_synthetic_golden(case_name, golden_check):
    data = SYNTHETIC_CASES[case_name]()
    result = charslit.slitdec(
        data["im"],
        data["pix_unc"],
        data["mask"],
        data["ycen"],
        data["slitcurve"],
        data["slitdeltas"],
        **data["kwargs"],
    )
    assert result["return_code"] == 0
    golden_check(f"synthetic_{case_name}", result)


def test_real_data_golden(real_data_files, golden_check):
    data = real_data_files
    result = charslit.slitdec(
        data["im"],
        data["pix_unc"],
        data["mask"],
        data["ycen"],
        data["slitcurve"],
        data["slitdeltas"],
        osample=data["osample"],
        lambda_sP=data["lambda_sP"],
        lambda_sL=data["lambda_sL"],
    )
    assert result["return_code"] == 0, f"slitdec failed on {data['filename']}"
    golden_check(f"real_{Path(data['filename']).stem}", result)
