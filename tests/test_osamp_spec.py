"""Regression tests for the osamp_spec (dispersion-direction oversampling) path.

The headline guarantee of the osamp_spec feature is that osamp_spec=1 reproduces
the pre-feature solver behaviour exactly. There is no pre-feature binary to diff
against any more, so instead we pin the osamp_spec=1 outputs of a deterministic
synthetic case to a committed golden reference. Any future refactor of the
solver that silently perturbs the s=1 path will trip this test.
"""

from pathlib import Path

import numpy as np
import pytest

import charslit

GOLDEN = Path(__file__).parent / "golden" / "osamp_spec1.npz"


def build_input():
    """Deterministic synthetic single-order image (mirrors simple_image_data)."""
    nrows, ncols, osample = 20, 50, 6

    x = np.arange(ncols)
    spectrum = np.exp(-0.5 * ((x - ncols / 2) / 5) ** 2) * 100 + 10

    ny = osample * (nrows + 1) + 1
    y = np.arange(ny) / osample - nrows / 2
    slitfunc = np.exp(-0.5 * (y / 2) ** 2)
    slitfunc /= slitfunc.sum() / osample

    im = np.zeros((nrows, ncols))
    for i in range(ncols):
        for j in range(nrows):
            slit_contrib = slitfunc[j * osample:(j + 1) * osample + 1].sum()
            im[j, i] = spectrum[i] * slit_contrib

    rng = np.random.RandomState(42)
    im += rng.normal(0, 1, im.shape)

    return {
        "im": im,
        "pix_unc": np.ones_like(im),
        "mask": np.ones(im.shape, dtype=np.uint8),
        "ycen": np.full(ncols, 0.5),
        "slitcurve": np.zeros((ncols, 3)),
        "slitdeltas": np.zeros(ny),
        "nrows": nrows,
        "ncols": ncols,
        "osample": osample,
        "ny": ny,
    }


def run_s1(data):
    return charslit.slitdec(
        data["im"],
        data["pix_unc"],
        data["mask"].copy(),
        data["ycen"],
        data["slitcurve"],
        data["slitdeltas"],
        osample=data["osample"],
        osamp_spec=1,
    )


def test_default_osamp_spec_is_one():
    """Omitting osamp_spec yields a coarse-grid spectrum and matches s=1."""
    data = build_input()
    default = charslit.slitdec(
        data["im"], data["pix_unc"], data["mask"].copy(), data["ycen"],
        data["slitcurve"], data["slitdeltas"], osample=data["osample"],
    )
    assert default["spectrum"].shape == (data["ncols"],)

    explicit = run_s1(data)
    np.testing.assert_array_equal(default["spectrum"], explicit["spectrum"])
    np.testing.assert_array_equal(default["slitfunction"], explicit["slitfunction"])


def test_osamp_spec_fine_grid_shape():
    """osamp_spec=s returns the spectrum on a grid of length ncols*s."""
    data = build_input()
    for s in (2, 3):
        result = charslit.slitdec(
            data["im"], data["pix_unc"], data["mask"].copy(), data["ycen"],
            data["slitcurve"], data["slitdeltas"], osample=data["osample"],
            osamp_spec=s,
        )
        assert result["return_code"] == 0
        assert result["spectrum"].shape == (data["ncols"] * s,)


@pytest.mark.skipif(not GOLDEN.exists(),
                    reason="golden reference missing; run `python tests/test_osamp_spec.py`")
def test_osamp_spec1_matches_golden():
    """Pin the s=1 solver outputs against the committed golden reference."""
    result = run_s1(build_input())
    ref = np.load(GOLDEN)
    for key in ("spectrum", "slitfunction", "model", "uncertainty"):
        np.testing.assert_allclose(
            result[key], ref[key], rtol=1e-9, atol=1e-9,
            err_msg=f"osamp_spec=1 {key} drifted from golden reference",
        )


def _write_golden():
    result = run_s1(build_input())
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        GOLDEN,
        spectrum=result["spectrum"],
        slitfunction=result["slitfunction"],
        model=result["model"],
        uncertainty=result["uncertainty"],
    )
    print(f"wrote golden reference: {GOLDEN}")


if __name__ == "__main__":
    _write_golden()
