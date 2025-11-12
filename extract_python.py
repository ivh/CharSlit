"""
Pure Python implementation of the slit decomposition algorithm.

This module provides a 1:1 translation of the C code in extract.c to Python,
using numpy for numerical calculations. The calculation steps are identical
to the original optimized C implementation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class XiRef:
    """Reference to detector pixel from subpixel (same as xi_ref in C)."""
    x: int = -1
    y: int = -1
    w: float = 0.0


@dataclass
class ZetaRef:
    """Reference to subpixel contributing to detector pixel (same as zeta_ref in C)."""
    x: int = -1
    iy: int = -1
    w: float = 0.0


def signum(a: float) -> int:
    """Return sign of a number: 1 for positive, -1 for negative, 0 for zero."""
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0


def quick_select_median(arr: np.ndarray) -> float:
    """
    Calculate median using quickselect algorithm.

    This is a 1:1 translation of the C implementation from Numerical Recipes.
    Note: This modifies the input array in place.

    Parameters
    ----------
    arr : np.ndarray
        Array of values (will be modified in place)

    Returns
    -------
    float
        The median value
    """
    n = len(arr)
    low = 0
    high = n - 1
    median = (low + high) // 2

    while True:
        if high <= low:  # One element only
            return arr[median]

        if high == low + 1:  # Two elements only
            if arr[low] > arr[high]:
                arr[low], arr[high] = arr[high], arr[low]
            return arr[median]

        # Find median of low, middle and high items; swap into position low
        middle = (low + high) // 2
        if arr[middle] > arr[high]:
            arr[middle], arr[high] = arr[high], arr[middle]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[middle] > arr[low]:
            arr[middle], arr[low] = arr[low], arr[middle]

        # Swap low item (now in position middle) into position (low+1)
        arr[middle], arr[low + 1] = arr[low + 1], arr[middle]

        # Nibble from each end towards middle, swapping items when stuck
        ll = low + 1
        hh = high

        while True:
            ll += 1
            while arr[low] > arr[ll]:
                ll += 1

            hh -= 1
            while arr[hh] > arr[low]:
                hh -= 1

            if hh < ll:
                break

            arr[ll], arr[hh] = arr[hh], arr[ll]

        # Swap middle item (in position low) back into correct position
        arr[low], arr[hh] = arr[hh], arr[low]

        # Re-set active partition
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1


def median_absolute_deviation(arr: np.ndarray) -> float:
    """
    Calculate median absolute deviation (MAD).

    Parameters
    ----------
    arr : np.ndarray
        Array of values (will be modified in place)

    Returns
    -------
    float
        The MAD value
    """
    median = quick_select_median(arr.copy())
    for i in range(len(arr)):
        arr[i] = abs(arr[i] - median)
    mad = quick_select_median(arr)
    return mad


def bandsol(a: np.ndarray, r: np.ndarray, n: int, nd: int) -> int:
    """
    Solve a sparse system of linear equations with band-diagonal matrix.

    Solves A * x = r where A is band-diagonal.

    Parameters
    ----------
    a : np.ndarray
        Band-diagonal matrix of shape (n, nd)
        The main diagonal is in a[:, nd//2]
        First lower subdiagonal in a[1:n, nd//2-1]
        First upper subdiagonal in a[0:n-1, nd//2+1]
    r : np.ndarray
        Right-hand side vector of shape (n,)
        On output, contains the solution
    n : int
        Number of equations
    nd : int
        Width of the band (must be odd)

    Returns
    -------
    int
        0 on success, -1 on failure
    """
    # Forward sweep
    for i in range(n - 1):
        aa = a[i, nd // 2]
        if aa == 0:
            aa = 1  # Avoid division by zero

        r[i] /= aa
        for j in range(nd):
            a[i, j] /= aa

        for j in range(1, min(nd // 2 + 1, n - i)):
            aa = a[i + j, nd // 2 - j]
            r[i + j] -= r[i] * aa
            for k in range(nd - j):
                a[i + j, k] -= a[i, k + j] * aa

    # Backward sweep
    aa = a[n - 1, nd // 2]
    if aa == 0:
        aa = 1
    r[n - 1] /= aa

    for i in range(n - 1, 0, -1):
        for j in range(1, min(nd // 2 + 1, i + 1)):
            r[i - j] -= r[i] * a[i - j, nd // 2 + j]
        r[i - 1] /= a[i - 1, nd // 2]

    aa = a[0, nd // 2]
    if aa == 0:
        aa = 1
    r[0] /= aa

    return 0


def xi_zeta_tensors(
    ncols: int,
    nrows: int,
    ny: int,
    ycen: np.ndarray,
    ycen_offset: np.ndarray,
    y_lower_lim: int,
    osample: int,
    slitdeltas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the Xi and Zeta tensors describing pixel-subpixel contributions.

    This function creates tensors that describe how each subpixel contributes
    to detector pixels, considering the curvature of the slit.

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    ny : int
        Size of slit function array: ny = osample * (nrows + 1) + 1
    ycen : np.ndarray
        Order centre line offset from pixel row boundary (ncols,)
    ycen_offset : np.ndarray
        Order image column shift (ncols,)
    y_lower_lim : int
        Number of detector pixels below the pixel containing the central line
    osample : int
        Subpixel oversampling factor
    slitdeltas : np.ndarray
        Slit curvature at each subpixel (ny,)

    Returns
    -------
    xi : np.ndarray
        Convolution tensor of shape (ncols, ny, 4) with dtype object containing XiRef
    zeta : np.ndarray
        Convolution tensor of shape (ncols, nrows, 3*(osample+1)) with dtype object containing ZetaRef
    m_zeta : np.ndarray
        Number of contributing elements in zeta for each pixel (ncols, nrows)
    """
    step = 1.0 / osample

    # Initialize xi - shape (ncols, ny, 4)
    xi = np.empty((ncols, ny, 4), dtype=object)
    for x in range(ncols):
        for iy in range(ny):
            for m in range(4):
                xi[x, iy, m] = XiRef()

    # Initialize zeta - shape (ncols, nrows, 3*(osample+1))
    max_zeta_z = 3 * (osample + 1)
    zeta = np.empty((ncols, nrows, max_zeta_z), dtype=object)
    m_zeta = np.zeros((ncols, nrows), dtype=np.int32)

    for x in range(ncols):
        for y in range(nrows):
            for ix in range(max_zeta_z):
                zeta[x, y, ix] = ZetaRef()

    # Construct the xi and zeta tensors
    for x in range(ncols):
        # Initialize subpixel indices for this column
        iy2 = osample - int(np.floor(ycen[x] * osample))
        iy1 = iy2 - osample

        # Handle partial subpixels cut by detector pixel rows
        d1 = np.fmod(ycen[x], step)
        if d1 == 0:
            d1 = step
        d2 = step - d1

        # Define initial distance from ycen
        dy = ycen[x] - np.floor((y_lower_lim + ycen[x]) / step) * step - step

        # Loop through detector pixels
        for y in range(nrows):
            iy1 += osample  # Bottom subpixel falling in row y
            iy2 += osample  # Top subpixel falling in row y
            dy -= step

            # Loop through subpixels in this row
            for iy in range(iy1, iy2 + 1):
                # Determine weight for this subpixel
                if iy == iy1:
                    w = d1
                elif iy == iy2:
                    w = d2
                else:
                    w = step

                dy += step
                delta = slitdeltas[iy]
                ix1 = int(delta)
                ix2 = ix1 + signum(delta)

                # Three cases: subpixel on bottom boundary, intermediate, or top boundary

                if iy == iy1:  # Case A: Subpixel entering detector row y
                    if ix1 < ix2:  # Shifts right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Upper right corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 3].x = xx
                            xi[x, iy, 3].y = yy
                            xi[x, iy, 3].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 3].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 3].w
                                m_zeta[xx, yy] += 1

                            # Upper left corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 2].x = xx
                            xi[x, iy, 2].y = yy
                            xi[x, iy, 2].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 2].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 2].w
                                m_zeta[xx, yy] += 1

                    elif ix1 > ix2:  # Shifts left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Upper left corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 2].x = xx
                            xi[x, iy, 2].y = yy
                            xi[x, iy, 2].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 2].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 2].w
                                m_zeta[xx, yy] += 1

                            # Upper right corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 3].x = xx
                            xi[x, iy, 3].y = yy
                            xi[x, iy, 3].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 3].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 3].w
                                m_zeta[xx, yy] += 1

                    else:  # No shift
                        xx = x + ix1
                        yy = y + ycen_offset[x] - ycen_offset[xx]
                        xi[x, iy, 2].x = xx
                        xi[x, iy, 2].y = yy
                        xi[x, iy, 2].w = w
                        if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and w > 0:
                            m = m_zeta[xx, yy]
                            zeta[xx, yy, m].x = x
                            zeta[xx, yy, m].iy = iy
                            zeta[xx, yy, m].w = w
                            m_zeta[xx, yy] += 1

                elif iy == iy2:  # Case C: Subpixel leaving detector row y
                    if ix1 < ix2:  # Shifts right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Bottom right corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 1].x = xx
                            xi[x, iy, 1].y = yy
                            xi[x, iy, 1].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 1].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 1].w
                                m_zeta[xx, yy] += 1

                            # Bottom left corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 0].x = xx
                            xi[x, iy, 0].y = yy
                            xi[x, iy, 0].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 0].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 0].w
                                m_zeta[xx, yy] += 1

                    elif ix1 > ix2:  # Shifts left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Bottom left corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 0].x = xx
                            xi[x, iy, 0].y = yy
                            xi[x, iy, 0].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 0].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 0].w
                                m_zeta[xx, yy] += 1

                            # Bottom right corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 1].x = xx
                            xi[x, iy, 1].y = yy
                            xi[x, iy, 1].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 1].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 1].w
                                m_zeta[xx, yy] += 1

                    else:  # No shift
                        xx = x + ix1
                        yy = y + ycen_offset[x] - ycen_offset[xx]
                        xi[x, iy, 0].x = xx
                        xi[x, iy, 0].y = yy
                        xi[x, iy, 0].w = w
                        if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and w > 0:
                            m = m_zeta[xx, yy]
                            zeta[xx, yy, m].x = x
                            zeta[xx, yy, m].iy = iy
                            zeta[xx, yy, m].w = w
                            m_zeta[xx, yy] += 1

                else:  # Case B: Subpixel fully inside detector row y
                    if ix1 < ix2:  # Shifts right
                        if x + ix1 >= 0 and x + ix2 < ncols:
                            # Bottom right corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 1].x = xx
                            xi[x, iy, 1].y = yy
                            xi[x, iy, 1].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 1].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 1].w
                                m_zeta[xx, yy] += 1

                            # Bottom left corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 0].x = xx
                            xi[x, iy, 0].y = yy
                            xi[x, iy, 0].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 0].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 0].w
                                m_zeta[xx, yy] += 1

                    elif ix1 > ix2:  # Shifts left
                        if x + ix2 >= 0 and x + ix1 < ncols:
                            # Bottom right corner
                            xx = x + ix2
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 1].x = xx
                            xi[x, iy, 1].y = yy
                            xi[x, iy, 1].w = abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 1].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 1].w
                                m_zeta[xx, yy] += 1

                            # Bottom left corner
                            xx = x + ix1
                            yy = y + ycen_offset[x] - ycen_offset[xx]
                            xi[x, iy, 0].x = xx
                            xi[x, iy, 0].y = yy
                            xi[x, iy, 0].w = w - abs(delta - ix1) * w
                            if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and xi[x, iy, 0].w > 0:
                                m = m_zeta[xx, yy]
                                zeta[xx, yy, m].x = x
                                zeta[xx, yy, m].iy = iy
                                zeta[xx, yy, m].w = xi[x, iy, 0].w
                                m_zeta[xx, yy] += 1

                    else:  # No shift
                        xx = x + ix2
                        yy = y + ycen_offset[x] - ycen_offset[xx]
                        xi[x, iy, 0].x = xx
                        xi[x, iy, 0].y = yy
                        xi[x, iy, 0].w = w
                        if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows and w > 0:
                            m = m_zeta[xx, yy]
                            zeta[xx, yy, m].x = x
                            zeta[xx, yy, m].iy = iy
                            zeta[xx, yy, m].w = w
                            m_zeta[xx, yy] += 1

    return xi, zeta, m_zeta


def extract(
    ncols: int,
    nrows: int,
    im: np.ndarray,
    pix_unc: np.ndarray,
    mask: np.ndarray,
    ycen: np.ndarray,
    slitdeltas: np.ndarray,
    osample: int,
    lambda_sP: float,
    lambda_sL: float,
    maxiter: int,
    slit_func_in: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract spectrum and slit illumination function for a curved slit.

    This is a 1:1 Python translation of the C extract function.

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    im : np.ndarray
        Image to be decomposed, shape (nrows, ncols)
    pix_unc : np.ndarray
        Individual pixel uncertainties, shape (nrows, ncols)
    mask : np.ndarray
        Initial mask for the swath (1=good, 0=bad), shape (nrows, ncols)
    ycen : np.ndarray
        Order centre line offset from pixel row boundary, shape (ncols,)
        Should only contain values between 0 and 1
    slitdeltas : np.ndarray
        Slit deltas for each subpixel along the slit, shape (ny,)
        where ny = osample * (nrows + 1) + 1
    osample : int
        Subpixel oversampling factor
    lambda_sP : float
        Smoothing parameter for the spectrum (can be zero)
    lambda_sL : float
        Smoothing parameter for the slit function (usually > 0)
    maxiter : int
        Maximum number of iterations
    slit_func_in : Optional[np.ndarray]
        Initial slit function guess, shape (ny,). If None, will be computed.

    Returns
    -------
    sP : np.ndarray
        Extracted spectrum, shape (ncols,)
    sL : np.ndarray
        Slit function, shape (ny,)
    model : np.ndarray
        Model image constructed from sP and sL, shape (nrows, ncols)
    unc : np.ndarray
        Spectrum uncertainties, shape (ncols,)
    info : np.ndarray
        Fit information: [success, cost, status, iterations, delta_x]
    """
    ftol = 1e-7
    success = 1
    status = 0
    cost = np.inf

    y_lower_lim = nrows // 2
    ny = osample * (nrows + 1) + 1

    # Calculate delta_x (maximum horizontal shift due to curvature)
    delta_x = 1 if lambda_sP == 0 else 1
    for iy in range(nrows):
        tmp = int(np.ceil(abs(slitdeltas[iy])))
        delta_x = max(delta_x, tmp)

    nx = 4 * delta_x + 1

    # Check if curvature is too large
    if nx > ncols:
        info = np.array([0, cost, -2, 0, delta_x])
        # Return empty arrays
        sP = np.zeros(ncols)
        sL = np.zeros(ny)
        model = np.zeros((nrows, ncols))
        unc = np.zeros(ncols)
        return sP, sL, model, unc, info

    # Separate integer and fractional parts of ycen
    ycen_offset = np.floor(ycen).astype(np.int32)
    ycen_frac = ycen - ycen_offset

    # Create xi and zeta tensors
    xi, zeta, m_zeta = xi_zeta_tensors(
        ncols, nrows, ny, ycen_frac, ycen_offset,
        y_lower_lim, osample, slitdeltas
    )

    # Initialize output arrays
    sP = np.ones(ncols)  # Initial guess for spectrum
    sL = np.ones(ny) if slit_func_in is None else slit_func_in.copy()
    model = np.zeros((nrows, ncols))
    unc = np.zeros(ncols)

    # Make a working copy of mask
    mask = mask.copy().astype(np.uint8)

    # Main iteration loop
    iter_count = 0
    while True:
        cost_old = cost

        # ========================================
        # Compute slit function sL
        # ========================================

        # Prepare arrays for band-diagonal system
        l_Aij = np.zeros((ny, 4 * osample + 1))
        l_bj = np.zeros(ny)

        # Fill in system of linear equations for slit function
        diag_tot = 0.0
        for iy in range(ny):
            for x in range(ncols):
                for n in range(4):
                    ww = xi[x, iy, n].w
                    if ww > 0:
                        xx = xi[x, iy, n].x
                        yy = xi[x, iy, n].y
                        if 0 <= xx < ncols and 0 <= yy < nrows:
                            if m_zeta[xx, yy] > 0:
                                for m in range(m_zeta[xx, yy]):
                                    xxx = zeta[xx, yy, m].x
                                    jy = zeta[xx, yy, m].iy
                                    www = zeta[xx, yy, m].w
                                    l_Aij[iy, jy - iy + 2 * osample] += (
                                        sP[xxx] * sP[x] * www * ww * mask[yy, xx]
                                    )
                                l_bj[iy] += im[yy, xx] * mask[yy, xx] * sP[x] * ww

            diag_tot += l_Aij[iy, 2 * osample]

        # Scale regularization parameter
        lambda_scaled = lambda_sL * diag_tot / ny

        # Add regularization (smoothing) terms
        l_Aij[0, 2 * osample] += lambda_scaled
        l_Aij[0, 2 * osample + 1] -= lambda_scaled

        for iy in range(1, ny - 1):
            l_Aij[iy, 2 * osample - 1] -= lambda_scaled
            l_Aij[iy, 2 * osample] += lambda_scaled * 2.0
            l_Aij[iy, 2 * osample + 1] -= lambda_scaled

        l_Aij[ny - 1, 2 * osample - 1] -= lambda_scaled
        l_Aij[ny - 1, 2 * osample] += lambda_scaled

        # Solve the system
        bandsol(l_Aij, l_bj, ny, 4 * osample + 1)

        # Normalize slit function
        sL = l_bj.copy()
        norm = np.sum(sL) / osample
        sL /= norm

        # ========================================
        # Compute spectrum sP
        # ========================================

        p_Aij = np.zeros((ncols, nx))
        p_bj = np.zeros(ncols)

        for x in range(ncols):
            for iy in range(ny):
                for n in range(4):
                    ww = xi[x, iy, n].w
                    if ww > 0:
                        xx = xi[x, iy, n].x
                        yy = xi[x, iy, n].y
                        if 0 <= xx < ncols and 0 <= yy < nrows:
                            if m_zeta[xx, yy] > 0:
                                for m in range(m_zeta[xx, yy]):
                                    xxx = zeta[xx, yy, m].x
                                    jy = zeta[xx, yy, m].iy
                                    www = zeta[xx, yy, m].w
                                    p_Aij[x, xxx - x + 2 * delta_x] += (
                                        sL[jy] * sL[iy] * www * ww * mask[yy, xx]
                                    )
                                p_bj[x] += im[yy, xx] * mask[yy, xx] * sL[iy] * ww

        # Add regularization if requested
        if lambda_sP > 0.0:
            norm = np.mean(sP)
            lambda_scaled = lambda_sP * norm

            p_Aij[0, 2 * delta_x] += lambda_scaled
            p_Aij[0, 2 * delta_x + 1] -= lambda_scaled

            for x in range(1, ncols - 1):
                p_Aij[x, 2 * delta_x - 1] -= lambda_scaled
                p_Aij[x, 2 * delta_x] += lambda_scaled * 2.0
                p_Aij[x, 2 * delta_x + 1] -= lambda_scaled

            p_Aij[ncols - 1, 2 * delta_x - 1] -= lambda_scaled
            p_Aij[ncols - 1, 2 * delta_x] += lambda_scaled

        # Solve the system
        bandsol(p_Aij, p_bj, ncols, nx)
        sP = p_bj.copy()

        # ========================================
        # Compute the model
        # ========================================

        model = np.zeros((nrows, ncols))
        for y in range(nrows):
            for x in range(ncols):
                for m in range(m_zeta[x, y]):
                    xx = zeta[x, y, m].x
                    iy = zeta[x, y, m].iy
                    ww = zeta[x, y, m].w
                    model[y, x] += sP[xx] * sL[iy] * ww

        # ========================================
        # Compare model and data
        # ========================================

        # Collect differences for MAD calculation
        diff_list = []
        for y in range(nrows):
            for x in range(delta_x, ncols - delta_x):
                if mask[y, x]:
                    tmp = model[y, x] - im[y, x]
                    diff_list.append(tmp)

        if len(diff_list) == 0:
            break

        diff_arr = np.array(diff_list)

        # Calculate cost
        cost = 0.0
        isum = 0
        for y in range(nrows):
            for x in range(delta_x, ncols - delta_x):
                if mask[y, x]:
                    tmp = model[y, x] - im[y, x]
                    tmp /= max(pix_unc[y, x], 1)
                    cost += tmp * tmp
                    isum += 1

        cost /= (isum - (ncols + ny))

        # Calculate MAD
        dev = median_absolute_deviation(diff_arr)
        dev *= 1.4826  # Conversion factor from MAD to STD

        # Adjust mask marking outliers
        for y in range(nrows):
            for x in range(delta_x, ncols - delta_x):
                if abs(model[y, x] - im[y, x]) < 40.0 * dev:
                    mask[y, x] = 1
                else:
                    mask[y, x] = 0

        # Check for convergence
        iter_count += 1
        if iter_count >= maxiter:
            status = -1
            success = 0
            break

        if np.isfinite(cost) and np.isfinite(cost_old):
            if cost_old - cost <= ftol:
                status = 1
                break

    # ========================================
    # Uncertainty estimate
    # ========================================

    unc = np.zeros(ncols)
    p_bj = np.zeros(ncols)
    p_Aij_0 = np.zeros(ncols)

    for y in range(nrows):
        for x in range(ncols):
            for m in range(m_zeta[x, y]):
                if mask[y, x]:
                    xx = zeta[x, y, m].x
                    iy = zeta[x, y, m].iy
                    ww = zeta[x, y, m].w
                    tmp = im[y, x] - model[y, x]
                    unc[xx] += tmp * tmp * ww
                    p_bj[xx] += ww
                    p_Aij_0[xx] += ww * ww

    for x in range(ncols):
        if p_bj[x] > 0:
            norm = p_bj[x] - p_Aij_0[x] / p_bj[x]
            if norm > 0:
                unc[x] = np.sqrt(unc[x] / norm * nrows)

    # Zero out edges
    for x in range(delta_x):
        sP[x] = unc[x] = 0
    for x in range(ncols - delta_x, ncols):
        sP[x] = unc[x] = 0

    info = np.array([success, cost, status, iter_count, delta_x])

    return sP, sL, model, unc, info


def create_spectral_model(
    ncols: int,
    nrows: int,
    osample: int,
    xi: np.ndarray,
    spec: np.ndarray,
    slitfunc: np.ndarray
) -> np.ndarray:
    """
    Create spectral model from spectrum and slit function.

    Parameters
    ----------
    ncols : int
        Number of columns
    nrows : int
        Number of rows
    osample : int
        Oversampling factor
    xi : np.ndarray
        Xi tensor from xi_zeta_tensors
    spec : np.ndarray
        Spectrum array, shape (ncols,)
    slitfunc : np.ndarray
        Slit function array, shape (ny,)

    Returns
    -------
    img : np.ndarray
        Model image, shape (nrows, ncols)
    """
    ny = (nrows + 1) * osample + 1
    img = np.zeros((nrows + 1, ncols))

    for x in range(ncols):
        for iy in range(ny):
            for m in range(4):
                pix_x = xi[x, iy, m].x
                pix_y = xi[x, iy, m].y
                pix_w = xi[x, iy, m].w
                if pix_x != -1 and pix_y != -1 and pix_w != 0:
                    img[pix_y, pix_x] += pix_w * spec[x] * slitfunc[iy]

    return img
