#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "slitdec.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define signum(a) (((a) > 0) ? 1 : ((a) < 0) ? -1 : 0)
#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef REGULARIZE_DIAGONAL
#define REGULARIZE_DIAGONAL 1
#endif

// Store important sizes in global variables to make access easier
// When calculating the proper indices
// When not checking the indices just the variables directly
#if DEBUG
int _ncols = 0;
int _ncols_fine = 0;
int _nrows = 0;
int _ny = 0;
int _nx = 0;
int _osample = 0;
int _osamp_spec = 0;
int _n = 0;
int _nd = 0;
#else
#define _ncols ncols
#define _ncols_fine ncols_fine
#define _nrows nrows
#define _ny ny
#define _nx nx
#define _osample osample
#define _osamp_spec osamp_spec
#define _n n
#define _nd nd
#endif

// Define the sizes of each array
#define MAX_ZETA_X (_ncols)
#define MAX_ZETA_Y (_nrows)
#define MAX_ZETA_Z (3 * ((_osample) + 1) * ((_osamp_spec) + 1))
#define MAX_ZETA (MAX_ZETA_X * MAX_ZETA_Y * MAX_ZETA_Z)
#define MAX_MZETA ((_ncols) * (_nrows))
#define MAX_XI ((_ncols_fine) * (_ny)*4)
#define MAX_CRV_X (_ncols)
#define MAX_CRV_Y (3)
#define MAX_CRV (MAX_CRV_X * MAX_CRV_Y)
#define MAX_A ((_n) * (_nd))
#define MAX_R (_n)
#define MAX_SP (_ncols_fine)
#define MAX_SL (_ny)
#define MAX_LAIJ_X (_ny)
#define MAX_LAIJ_Y (4 * (_osample) + 1)
#define MAX_LAIJ (MAX_LAIJ_X * MAX_LAIJ_Y)
#define MAX_PAIJ_X (_ncols_fine)
#define MAX_PAIJ_Y (_nx)
#define MAX_PAIJ (MAX_PAIJ_X * MAX_PAIJ_Y)
#define MAX_LBJ (_ny)
#define MAX_PBJ (_ncols_fine)
#define MAX_IM ((_ncols) * (_nrows))

// If we want to check the index use functions to represent the index
// Otherwise a simpler define will do, which should be faster ?
#if DEBUG
static long zeta_index(long x, long y, long z)
{
    long i = z + y * MAX_ZETA_Z + x * MAX_ZETA_Z * _nrows;
    if ((i < 0) | (i >= MAX_ZETA))
    {
        printf("INDEX OUT OF BOUNDS. Zeta[%li, %li, %li]\n", x, y, z);
        return 0;
    }
    return i;
}

static long mzeta_index(long x, long y)
{
    long i = y + x * _nrows;
    if ((i < 0) | (i >= MAX_MZETA))
    {
        printf("INDEX OUT OF BOUNDS. Mzeta[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long xi_index(long x, long y, long z)
{
    long i = z + 4 * y + _ny * 4 * x;
    if ((i < 0) | (i >= MAX_XI))
    {
        printf("INDEX OUT OF BOUNDS. Xi[%li, %li, %li]\n", x, y, z);
        return 0;
    }
    return i;
}

static long curve_index(long x, long y)
{
    long i = ((x)*3 + (y));
    if ((i < 0) | (i >= MAX_CRV))
    {
        printf("INDEX OUT OF BOUNDS. PSF[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long a_index(long x, long y)
{
    long i = _n * y + x;
    if ((i < 0) | (i >= MAX_A))
    {
        printf("INDEX OUT OF BOUNDS. a[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long r_index(long i)
{
    if ((i < 0) | (i >= MAX_R))
    {
        printf("INDEX OUT OF BOUNDS. r[%li]\n", i);
        return 0;
    }
    return i;
}

static long sp_index(long i)
{
    if ((i < 0) | (i >= MAX_SP))
    {
        printf("INDEX OUT OF BOUNDS. sP[%li]\n", i);
        return 0;
    }
    return i;
}

static long laij_index(long x, long y)
{
    long i = ((y)*_ny) + (x);
    if ((i < 0) | (i >= MAX_LAIJ))
    {
        printf("INDEX OUT OF BOUNDS. l_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long paij_index(long x, long y)
{
    long i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_PAIJ))
    {
        printf("INDEX OUT OF BOUNDS. p_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long lbj_index(long i)
{
    if ((i < 0) | (i >= MAX_LBJ))
    {
        printf("INDEX OUT OF BOUNDS. l_bj[%li]\n", i);
        return 0;
    }
    return i;
}

static long pbj_index(long i)
{
    if ((i < 0) | (i >= MAX_PBJ))
    {
        printf("INDEX OUT OF BOUNDS. p_bj[%li]\n", i);
        return 0;
    }
    return i;
}

static long im_index(long x, long y)
{
    long i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_IM))
    {
        printf("INDEX OUT OF BOUNDS. im[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long sl_index(long i)
{
    if ((i < 0) | (i >= MAX_SL))
    {
        printf("INDEX OUT OF BOUNDS. sL[%li]\n", i);
        return 0;
    }
    return i;
}
#else
#define zeta_index(x, y, z) ((z) + (y)*MAX_ZETA_Z + (x)*MAX_ZETA_Z * _nrows)
#define mzeta_index(x, y) ((y) + (x)*_nrows)
#define xi_index(x, y, z) ((z) + 4 * (y) + _ny * 4 * (x))
#define curve_index(x, y) ((x)*6 + (y))
#define a_index(x, y) ((y)*n + (x))
#define r_index(i) (i)
#define sp_index(i) (i)
#define laij_index(x, y) ((y)*ny) + (x)
#define paij_index(x, y) ((y)*ncols_fine) + (x)
#define lbj_index(i) (i)
#define pbj_index(i) (i)
#define im_index(x, y) ((y)*ncols) + (x)
#define sl_index(i) (i)
#endif

int bandsol(double *a, double *r, int n, int nd)
{
    /*
    bandsol solves a sparse system of linear equations with band-diagonal matrix.
    Band is assumed to be symmetric relative to the main diaginal.

    ..math:

        A * x = r

    Parameters
    ----------
    a : double array of shape [n,nd]
        The left-hand-side of the equation system
        The main diagonal should be in a(*,nd/2),
        the first lower subdiagonal should be in a(1:n-1,nd/2-1),
        the first upper subdiagonal is in a(0:n-2,nd/2+1) etc.
        For example:
                / 0 0 X X X \
                | 0 X X X X |
                | X X X X X |
                | X X X X X |
            A = | X X X X X |
                | X X X X X |
                | X X X X X |
                | X X X X 0 |
                \ X X X 0 0 /
    r : double array of shape [n]
        the right-hand-side of the equation system
    n : int
        The number of equations
    nd : int
        The width of the band (3 for tri-diagonal system). Must be an odd number.

    Returns
    -------
    code : int
        0 on success, -1 on incorrect size of "a" and -4 on degenerate matrix.
    */
    double aa;
    int i, j, k;

#if DEBUG
    _n = n;
    _nd = nd;
#endif

    /* Forward sweep */
    for (i = 0; i < n - 1; i++)
    {
        aa = a[a_index(i, nd / 2)];
#if DEBUG
        if (aa == 0)
        {
            printf("1, index: %i, %i\n", i, nd / 2);
            aa = 1;
        }
#endif
        r[r_index(i)] /= aa;
        for (j = 0; j < nd; j++)
            a[a_index(i, j)] /= aa;
        for (j = 1; j < min(nd / 2 + 1, n - i); j++)
        {
            aa = a[a_index(i + j, nd / 2 - j)];
            r[r_index(i + j)] -= r[r_index(i)] * aa;
            for (k = 0; k < nd - j; k++)
                a[a_index(i + j, k)] -= a[a_index(i, k + j)] * aa;
        }
    }

    /* Backward sweep */
    aa = a[a_index(n - 1, nd / 2)];
#if DEBUG
    if (aa == 0)
    {
        printf("3, index: %i, %i\n", 0, nd / 2);
        aa = 1;
    }
#endif
    r[r_index(n - 1)] /= aa;
    for (i = n - 1; i > 0; i--)
    {
        for (j = 1; j <= min(nd / 2, i); j++)
            r[r_index(i - j)] -= r[r_index(i)] * a[a_index(i - j, nd / 2 + j)];
        r[r_index(i - 1)] /= a[a_index(i - 1, nd / 2)];
    }

    aa = a[a_index(0, nd / 2)];
#if DEBUG
    if (aa == 0)
    {
        printf("4, index: %i, %i\n", 0, nd / 2);
        aa = 1;
    }
#endif
    r[r_index(0)] /= aa;
    return 0;
}

// This is the faster median determination method.
// Algorithm from Numerical recipes in C of 1992
int xi_zeta_tensors(
    int ncols,
    int nrows,
    int ny,
    int osamp_spec,
    double *ycen,
    int *ycen_offset,
    int y_lower_lim,
    int osample,
    double *slitcurve,
    double *slitdeltas,
    xi_ref *xi,
    zeta_ref *zeta,
    int *m_zeta)
{
    int ncols_fine = ncols * osamp_spec;
    /*
    Create the Xi and Zeta tensors, that describe the contribution of each pixel to the subpixels of the image,
    Considering the curvature of the slit.

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    ny : int
        Size of the slit function array: ny = osample * (nrows + 1) + 1
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary
    ycen_offsets : int array of shape (ncols,)
        Order image column shift
    y_lower_lim : int
        Number of detector pixels below the pixel containing the central line ycen
    osample : int
        Subpixel ovsersampling factor
    slitcurve : double array of shape (ncols, 3)
        Parabolic fit to the slit image curvature.
        For column d_x = slitcurve[ncols][0] +  slitcurve[ncols][1] *d_y + slitcurve[ncols][2] *d_y^2,
        where d_y is the offset from the central line ycen.
        Thus central subpixel of omega[x][y'][delta_x][iy'] does not stick out of column x.
    xi : (out) xi_ref array of shape (ncols, ny, 4)
        Convolution tensor telling the coordinates of detector
        pixels on which {x, iy} element falls and the corresponding projections.
    zeta : (out) zeta_ref array of shape (ncols, nrows, 3 * (osample + 1))
        Convolution tensor telling the coordinates of subpixels {x, iy} contributing
        to detector pixel {x, y}.
    m_zeta : (out) int array of shape (ncols, nrows)
        The actual number of contributing elements in zeta for each pixel

    Returns
    -------
    code : int
        0 on success, -1 on failure
    */
    int x, xx, y, yy, ix, iy, iy1, iy2, m;
    int x_fine;
    double step, delta, dy, w, d1, d2;
    double x_off_frac, frac_fine, w_xL, w_xR;
    int xx_left, xx_right, corner_left, corner_right;

    step = 1.e0 / osample;
    frac_fine = 1.e0 / osamp_spec;

    /* Clean xi */
    for (x_fine = 0; x_fine < ncols_fine; x_fine++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (m = 0; m < 4; m++)
            {
                xi[xi_index(x_fine, iy, m)].x = -1;
                xi[xi_index(x_fine, iy, m)].y = -1;
                xi[xi_index(x_fine, iy, m)].w = 0.;
            }
        }
    }

    /* Clean zeta */
    for (x = 0; x < ncols; x++)
    {
        for (y = 0; y < nrows; y++)
        {
            m_zeta[mzeta_index(x, y)] = 0;
            for (ix = 0; ix < MAX_ZETA_Z; ix++)
            {
                zeta[zeta_index(x, y, ix)].x = -1;
                zeta[zeta_index(x, y, ix)].iy = -1;
                zeta[zeta_index(x, y, ix)].w = 0.;
            }
        }
    }

    /*
    Construct the xi and zeta tensors. They contain pixel references and contribution.
    values going from a given subpixel to other pixels (xi) and coming from other subpixels
    to a given detector pixel (zeta).
    Note, that xi and zeta are used in the equations for sL, sP and for the model but they
    do not involve the data, only the geometry. Thus it can be pre-computed once.
    */
    for (x_fine = 0; x_fine < ncols_fine; x_fine++)
    {
        x = x_fine / osamp_spec;
        x_off_frac = (double)(x_fine - x * osamp_spec) / (double)osamp_spec;

        iy2 = osample - floor(ycen[x] * osample);
        iy1 = iy2 - osample;

        d1 = fmod(ycen[x], step);
        if (d1 == 0)
            d1 = step;
        d2 = step - d1;

        dy = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;

        for (y = 0; y < nrows; y++)
        {
            iy1 += osample;
            iy2 += osample;
            dy -= step;
            for (iy = iy1; iy <= iy2; iy++)
            {
                if (iy == iy1)
                    w = d1;
                else if (iy == iy2)
                    w = d2;
                else
                    w = step;
                dy += step;
                {
                    double t = dy - ycen[x];
                    delta = t * (slitcurve[curve_index(x, 1)] +
                            t * (slitcurve[curve_index(x, 2)] +
                            t * (slitcurve[curve_index(x, 3)] +
                            t * (slitcurve[curve_index(x, 4)] +
                            t *  slitcurve[curve_index(x, 5)]))))
                            + slitdeltas[iy];
                }

                if (osamp_spec == 1)
                {
                    /* Original algorithm verbatim — preserves slot assignment
                       and accumulation order to guarantee bit-identity with
                       the pre-osamp_spec implementation. */
                    int ix1, ix2;
                    ix1 = delta;
                    ix2 = ix1 + signum(delta);

                    if (iy == iy1) /* Case A: entering row y */
                    {
                        if (ix1 < ix2)
                        {
                            if (x + ix1 >= 0 && x + ix2 < ncols)
                            {
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 3)].x = xx;
                                xi[xi_index(x_fine, iy, 3)].y = yy;
                                xi[xi_index(x_fine, iy, 3)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 3)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 3)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 2)].x = xx;
                                xi[xi_index(x_fine, iy, 2)].y = yy;
                                xi[xi_index(x_fine, iy, 2)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 2)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 2)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else if (ix1 > ix2)
                        {
                            if (x + ix2 >= 0 && x + ix1 < ncols)
                            {
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 2)].x = xx;
                                xi[xi_index(x_fine, iy, 2)].y = yy;
                                xi[xi_index(x_fine, iy, 2)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 2)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 2)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 3)].x = xx;
                                xi[xi_index(x_fine, iy, 3)].y = yy;
                                xi[xi_index(x_fine, iy, 3)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 3)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 3)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x_fine, iy, 2)].x = xx;
                            xi[xi_index(x_fine, iy, 2)].y = yy;
                            xi[xi_index(x_fine, iy, 2)].w = w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x_fine;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (iy == iy2) /* Case C: leaving row y */
                    {
                        if (ix1 < ix2)
                        {
                            if (x + ix1 >= 0 && x + ix2 < ncols)
                            {
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 1)].x = xx;
                                xi[xi_index(x_fine, iy, 1)].y = yy;
                                xi[xi_index(x_fine, iy, 1)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 1)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 1)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 0)].x = xx;
                                xi[xi_index(x_fine, iy, 0)].y = yy;
                                xi[xi_index(x_fine, iy, 0)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 0)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 0)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else if (ix1 > ix2)
                        {
                            if (x + ix2 >= 0 && x + ix1 < ncols)
                            {
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 0)].x = xx;
                                xi[xi_index(x_fine, iy, 0)].y = yy;
                                xi[xi_index(x_fine, iy, 0)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 0)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 0)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 1)].x = xx;
                                xi[xi_index(x_fine, iy, 1)].y = yy;
                                xi[xi_index(x_fine, iy, 1)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 1)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 1)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x_fine, iy, 0)].x = xx;
                            xi[xi_index(x_fine, iy, 0)].y = yy;
                            xi[xi_index(x_fine, iy, 0)].w = w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x_fine;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else /* Case B: interior of row y */
                    {
                        if (ix1 < ix2)
                        {
                            if (x + ix1 >= 0 && x + ix2 < ncols)
                            {
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 1)].x = xx;
                                xi[xi_index(x_fine, iy, 1)].y = yy;
                                xi[xi_index(x_fine, iy, 1)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 1)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 1)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 0)].x = xx;
                                xi[xi_index(x_fine, iy, 0)].y = yy;
                                xi[xi_index(x_fine, iy, 0)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 0)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 0)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else if (ix1 > ix2)
                        {
                            if (x + ix2 >= 0 && x + ix1 < ncols)
                            {
                                xx = x + ix2;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 1)].x = xx;
                                xi[xi_index(x_fine, iy, 1)].y = yy;
                                xi[xi_index(x_fine, iy, 1)].w = fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 1)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 1)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                                xx = x + ix1;
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, 0)].x = xx;
                                xi[xi_index(x_fine, iy, 0)].y = yy;
                                xi[xi_index(x_fine, iy, 0)].w = w - fabs(delta - ix1) * w;
                                if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x_fine, iy, 0)].w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x_fine, iy, 0)].w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                        else
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x_fine, iy, 0)].x = xx;
                            xi[xi_index(x_fine, iy, 0)].y = yy;
                            xi[xi_index(x_fine, iy, 0)].w = w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x_fine;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                }
                else
                {
                    /* General osamp_spec > 1 path: fine subpixel occupies
                       [x + pos_L, x + pos_L + frac_fine) where
                       pos_L = x_off_frac + delta.  It straddles an integer
                       boundary iff frac_L + frac_fine > 1. */
                    double pos_L = x_off_frac + delta;
                    int    dx_left = (int)floor(pos_L);
                    double frac_L = pos_L - (double)dx_left;
                    int    dx_right = dx_left;
                    if (frac_L + frac_fine > 1.0)
                    {
                        dx_right = dx_left + 1;
                        w_xL = 1.0 - frac_L;
                        w_xR = frac_L + frac_fine - 1.0;
                    }
                    else
                    {
                        w_xL = frac_fine;
                        w_xR = 0.0;
                    }
                    xx_left  = x + dx_left;
                    xx_right = x + dx_right;

                    if (iy == iy1)
                    {
                        corner_left  = 2;
                        corner_right = 3;
                    }
                    else
                    {
                        corner_left  = 0;
                        corner_right = 1;
                    }

                    int boundary_ok = (xx_left != xx_right)
                        ? (xx_left >= 0 && xx_right < ncols)
                        : (xx_left >= 0 && xx_left < ncols);
                    if (boundary_ok)
                    {
                        xx = xx_left;
                        if (xx >= 0 && xx < ncols && w_xL > 0)
                        {
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x_fine, iy, corner_left)].x = xx;
                            xi[xi_index(x_fine, iy, corner_left)].y = yy;
                            xi[xi_index(x_fine, iy, corner_left)].w = w_xL * w;
                            if (yy >= 0 && yy < nrows && w_xL * w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x_fine;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w_xL * w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                        if (w_xR > 0)
                        {
                            xx = xx_right;
                            if (xx >= 0 && xx < ncols)
                            {
                                yy = y + ycen_offset[x] - ycen_offset[xx];
                                xi[xi_index(x_fine, iy, corner_right)].x = xx;
                                xi[xi_index(x_fine, iy, corner_right)].y = yy;
                                xi[xi_index(x_fine, iy, corner_right)].w = w_xR * w;
                                if (yy >= 0 && yy < nrows && w_xR * w > 0)
                                {
                                    m = m_zeta[mzeta_index(xx, yy)];
                                    zeta[zeta_index(xx, yy, m)].x = x_fine;
                                    zeta[zeta_index(xx, yy, m)].iy = iy;
                                    zeta[zeta_index(xx, yy, m)].w = w_xR * w;
                                    m_zeta[mzeta_index(xx, yy)]++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int slitdec(        int ncols,
                    int nrows,
                    double *im,
                    double *pix_unc,
                    unsigned char *mask,
                    double *ycen,
                    double *slitcurve,
                    double *slitdeltas,
                    int osample,
                    int osamp_spec,
                    double lambda_sP,
                    double lambda_sL,
                    double lambda_fringe,
                    int maxiter,
                    double kappa,
                    double *sP,
                    double *sL,
                    double *model,
                    double *unc,
                    double *info)
{
    /*
    Extract the spectrum and slit illumination function for a curved slit

    This function does not assign or free any memory,
    therefore all working arrays are passed as parameters.
    The contents of which will be overriden however

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    im : double array of shape (nrows, ncols)
        Image to be decomposed
    pix_unc : double array of shape (nrows, ncols)
        Individual pixel uncertainties. Set to zero if unknown.
    mask : byte array of shape (nrows, ncols)
        Initial and final mask for the swath, both in and output
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary.
        Should only contain values between 0 and 1.
    slitcurve : double array of shape (ncols, 3)
        Slit curvature parameters for each point along the spectrum
    slitdeltas : double array of shape (nrows, ncols)
        Slit deltas for each point along the slit
    osample : int
        Subpixel ovsersampling factor
    lambda_sP : double
        Smoothing parameter for the spectrum, could be zero
    lambda_sL : double
        Smoothing parameter for the slit function, usually > 0
    sP : (out) double array of shape (ncols,)
        Spectrum resulting from decomposition
    sL : (out) double array of shape (ny,)
        Slit function resulting from decomposition
    model : (out) double array of shape (ncols, nrows)
        Model constructed from sp and sf
    unc : (out) double array of shape (ncols,)
        Spectrum uncertainties based on data - model and pix_unc
    info : (out) double array of shape (5,)
        Returns information about the fit results
    Returns
    -------
    code : int
        0 on success, -1 on failure (see also bandsol)
    */
    int x, xx, xxx, y, yy, iy, jy, n, m, nx, ny;
    int ncols_fine, delta_x_fine;
    double norm, dev, lambda, diag_tot, ww, www;
    double cost_old, ftol, tmp;
    int iter, delta_x;
    unsigned int isum;
    int *ycen_offset;
    int y_lower_lim = nrows / 2;

    if (osamp_spec < 1) osamp_spec = 1;
    ncols_fine = ncols * osamp_spec;

    // For the solving of the equation system
    double *l_Aij, *l_bj, *p_Aij, *p_bj;

    // For the geometry
    xi_ref *xi;
    zeta_ref *zeta;
    int *m_zeta;

    // The Optimization results
    double success, status, cost;

    // maxiter = 20; // Maximum number of iterations
    ftol = 1e-7;  // Maximum cost difference between two iterations to stop convergence
    success = 1;
    status = 0;

    cost = INFINITY;
    ny = osample * (nrows + 1) + 1; /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */

#if DEBUG
    _ncols = ncols;
    _ncols_fine = ncols_fine;
    _nrows = nrows;
    _ny = ny;
    _osample = osample;
    _osamp_spec = osamp_spec;
    printf("ncols: %d, ncols_fine: %d, nrows: %d, ny: %d, osample: %d, osamp_spec: %d\n",
           _ncols, _ncols_fine, _nrows, _ny, _osample, _osamp_spec);
#endif

    // If we want to smooth the spectrum we need at least delta_x = 1
    // Otherwise delta_x = 0 works if there is no curvature
    delta_x = lambda_sP == 0 ? 0 : 1;
    for (x = 0; x < ncols; x++)
    {
        for (y = -y_lower_lim; y < nrows - y_lower_lim + 1; y++)
        {
            double y2 = y * y;
            double y3 = y2 * y;
            double y4 = y3 * y;
            double y5 = y4 * y;
            tmp = ceil(fabs(y * slitcurve[curve_index(x, 1)] +
                           y2 * slitcurve[curve_index(x, 2)] +
                           y3 * slitcurve[curve_index(x, 3)] +
                           y4 * slitcurve[curve_index(x, 4)] +
                           y5 * slitcurve[curve_index(x, 5)]));
            delta_x = max(delta_x, tmp);
        }
    }

    // Account for additional shift from slitdeltas
    for (int iy = 0; iy < ny; iy++)
    {
        tmp = ceil(fabs(slitdeltas[iy]));
        delta_x = max(delta_x, tmp);
    }

    /* Full band width of the spectrum normal-matrix on the fine grid.
       Derived from: two fine bins can both land in the same detector column
       iff |x_fine - xxx_fine| < osamp_spec * (2*delta_x + 1), giving
       nx = 2*(osamp_spec*(2*delta_x+1) - 1) + 1 = 4*delta_x*osamp_spec + 2*osamp_spec - 1.
       For osamp_spec=1 this reduces to 4*delta_x + 1, matching the original. */
    nx = 4 * delta_x * osamp_spec + 2 * osamp_spec - 1;
    delta_x_fine = (nx - 1) / 2;

#if DEBUG
    _nx = nx;
#endif

    // The curvature is larger than the fine-grid matrix dimension.
    // Usually that means that the curvature is messed up.
    if (nx > ncols_fine)
    {
        info[0] = 0;    //failed
        info[1] = cost; //INFINITY
        info[2] = -2;   // curvature to large
        info[3] = 0;
        info[4] = delta_x;
        return -1;
    }

    l_Aij = malloc(MAX_LAIJ * sizeof(double));
    p_Aij = malloc(MAX_PAIJ * sizeof(double));
    l_bj = malloc(MAX_LBJ * sizeof(double));
    p_bj = malloc(MAX_PBJ * sizeof(double));
    xi = malloc(MAX_XI * sizeof(xi_ref));
    zeta = malloc(MAX_ZETA * sizeof(zeta_ref));
    m_zeta = malloc(MAX_MZETA * sizeof(int));
    ycen_offset = malloc(ncols * sizeof(int));

        // remove integer values from ycen, put into ycen_offset
    for (x = 0; x < ncols; x++)
    {
        ycen_offset[x] = ycen[x];
        ycen[x] = ycen[x] - ycen_offset[x];
    }
    
    xi_zeta_tensors(ncols, nrows, ny, osamp_spec, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas, xi, zeta, m_zeta);

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    do
    {
        // Save the total cost (chi-square) from the previous iteration
        cost_old = cost;

        /* Compute slit function sL */

        /* Prepare the RHS and the matrix */
        for (iy = 0; iy < MAX_LBJ; iy++)
            l_bj[lbj_index(iy)] = 0.e0; /* Clean RHS */
        for (iy = 0; iy < MAX_LAIJ; iy++)
            l_Aij[iy] = 0;

        /* Fill in SLE arrays for slit function.
           Outer column loop is over *fine* spectrum bins since xi/zeta are
           indexed by the fine bin index. sL length and band structure are
           unaffected by osamp_spec. */
        diag_tot = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
            for (x = 0; x < ncols_fine; x++)
            {
                for (n = 0; n < 4; n++)
                {
                    ww = xi[xi_index(x, iy, n)].w;
                    if (ww > 0)
                    {
                        xx = xi[xi_index(x, iy, n)].x;
                        yy = xi[xi_index(x, iy, n)].y;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows)
                        {
                            if (m_zeta[mzeta_index(xx, yy)] > 0)
                            {
                                for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                                {
                                    xxx = zeta[zeta_index(xx, yy, m)].x;
                                    jy = zeta[zeta_index(xx, yy, m)].iy;
                                    www = zeta[zeta_index(xx, yy, m)].w;
                                    l_Aij[laij_index(iy, jy - iy + 2 * osample)] += sP[sp_index(xxx)] * sP[sp_index(x)] * www * ww * mask[im_index(xx, yy)];
                                }
                                l_bj[lbj_index(iy)] += im[im_index(xx, yy)] * mask[im_index(xx, yy)] * sP[sp_index(x)] * ww;
                            }
                        }
                    }
                }
            }
            diag_tot += l_Aij[laij_index(iy, 2 * osample)];
        }

        /* Scale regularization parameters */
        lambda = lambda_sL * diag_tot / ny;

        /* Add regularization parts for the SLE matrix */

        l_Aij[laij_index(0, 2 * osample)] += lambda;     /* Main diagonal  */
        l_Aij[laij_index(0, 2 * osample + 1)] -= lambda; /* Upper diagonal */
        for (iy = 1; iy < ny - 1; iy++)
        {
            l_Aij[laij_index(iy, 2 * osample - 1)] -= lambda;    /* Lower diagonal */
            l_Aij[laij_index(iy, 2 * osample)] += lambda * 2.e0; /* Main diagonal  */
            l_Aij[laij_index(iy, 2 * osample + 1)] -= lambda;    /* Upper diagonal */
        }
        l_Aij[laij_index(ny - 1, 2 * osample - 1)] -= lambda; /* Lower diagonal */
        l_Aij[laij_index(ny - 1, 2 * osample)] += lambda;     /* Main diagonal  */

#if REGULARIZE_DIAGONAL
        /* Regularize diagonal to prevent singular matrix from fully masked rows */
        {
            double max_diag = 0.0;
            for (iy = 0; iy < ny; iy++)
            {
                if (l_Aij[laij_index(iy, 2 * osample)] > max_diag)
                    max_diag = l_Aij[laij_index(iy, 2 * osample)];
            }
            if (max_diag > 0.0)
            {
                double min_diag = max_diag * 1.0e-10;
                for (iy = 0; iy < ny; iy++)
                {
                    if (l_Aij[laij_index(iy, 2 * osample)] < min_diag)
                        l_Aij[laij_index(iy, 2 * osample)] = min_diag;
                }
            }
        }
#endif

        /* Solve the system of equations */
        bandsol(l_Aij, l_bj, MAX_LAIJ_X, MAX_LAIJ_Y);

        /* Normalize the slit function */

        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
            sL[sl_index(iy)] = l_bj[lbj_index(iy)];
            norm += sL[sl_index(iy)];
        }
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[sl_index(iy)] /= norm;

        /* Compute spectrum sP */
        for (x = 0; x < MAX_PBJ; x++)
            p_bj[pbj_index(x)] = 0;
        for (x = 0; x < MAX_PAIJ; x++)
            p_Aij[x] = 0;

        for (x = 0; x < ncols_fine; x++)
        {
            for (iy = 0; iy < ny; iy++)
            {
                for (n = 0; n < 4; n++)
                {
                    ww = xi[xi_index(x, iy, n)].w;
                    if (ww > 0)
                    {
                        xx = xi[xi_index(x, iy, n)].x;
                        yy = xi[xi_index(x, iy, n)].y;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows)
                        {
                            if (m_zeta[mzeta_index(xx, yy)] > 0)
                            {
                                for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                                {
                                    xxx = zeta[zeta_index(xx, yy, m)].x;
                                    jy = zeta[zeta_index(xx, yy, m)].iy;
                                    www = zeta[zeta_index(xx, yy, m)].w;
                                    p_Aij[paij_index(x, xxx - x + delta_x_fine)] += sL[sl_index(jy)] * sL[sl_index(iy)] * www * ww * mask[im_index(xx, yy)];
                                }
                                p_bj[pbj_index(x)] += im[im_index(xx, yy)] * mask[im_index(xx, yy)] * sL[sl_index(iy)] * ww;
                            }
                        }
                    }
                }
            }
        }

        if (lambda_sP > 0.e0)
        {
            lambda = lambda_sP;

            p_Aij[paij_index(0, delta_x_fine)] += lambda;     /* Main diagonal  */
            p_Aij[paij_index(0, delta_x_fine + 1)] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols_fine - 1; x++)
            {
                p_Aij[paij_index(x, delta_x_fine - 1)] -= lambda;    /* Lower diagonal */
                p_Aij[paij_index(x, delta_x_fine)] += lambda * 2.e0; /* Main diagonal  */
                p_Aij[paij_index(x, delta_x_fine + 1)] -= lambda;    /* Upper diagonal */
            }
            p_Aij[paij_index(ncols_fine - 1, delta_x_fine - 1)] -= lambda; /* Lower diagonal */
            p_Aij[paij_index(ncols_fine - 1, delta_x_fine)] += lambda;     /* Main diagonal  */
        }

        /* Selective regularizer targeting the osamp-period fringe mode.
           Penalty: lambda_fringe * sum_blocks sum_i (sP[i] - mean_block)^2.
           The operator L = I - (1/s)*J per block of size s is symmetric and
           idempotent, so L^T L = L; we add lambda_fringe * L directly to
           the normal matrix. The row-sum is zero, so the coarse-averaged
           component is untouched and the penalty is surgical on the
           osamp-period null direction. */
        if (lambda_fringe > 0.e0 && osamp_spec > 1)
        {
            double s_inv = 1.0 / (double)osamp_spec;
            double diag_add = lambda_fringe * (1.0 - s_inv);
            double off_add  = -lambda_fringe * s_inv;
            for (int c = 0; c < ncols; c++)
            {
                for (int i = 0; i < osamp_spec; i++)
                {
                    int xi_fine = c * osamp_spec + i;
                    for (int j = 0; j < osamp_spec; j++)
                    {
                        int xj_fine = c * osamp_spec + j;
                        int k = xj_fine - xi_fine + delta_x_fine;
                        if (i == j)
                            p_Aij[paij_index(xi_fine, k)] += diag_add;
                        else
                            p_Aij[paij_index(xi_fine, k)] += off_add;
                    }
                }
            }
        }

#if REGULARIZE_DIAGONAL
        /* Regularize diagonal to prevent singular matrix from fully masked columns.
           When a column has no valid data (all pixels masked), the corresponding
           row of the matrix is zero, causing division by zero in bandsol.
           We add a small regularization to the diagonal to make it non-singular.
           The resulting spectrum value for masked columns will be ~0 (from p_bj[x]/diag). */
        {
            double max_diag = 0.0;
            for (x = 0; x < ncols_fine; x++)
            {
                if (p_Aij[paij_index(x, delta_x_fine)] > max_diag)
                    max_diag = p_Aij[paij_index(x, delta_x_fine)];
            }
            if (max_diag > 0.0)
            {
                double min_diag = max_diag * 1.0e-10;
                for (x = 0; x < ncols_fine; x++)
                {
                    if (p_Aij[paij_index(x, delta_x_fine)] < min_diag)
                        p_Aij[paij_index(x, delta_x_fine)] = min_diag;
                }
            }
        }
#endif

        /* Solve the system of equations */
        bandsol(p_Aij, p_bj, MAX_PAIJ_X, MAX_PAIJ_Y);

        for (x = 0; x < ncols_fine; x++)
            sP[sp_index(x)] = p_bj[pbj_index(x)];

        /* Compute the model */
        for (x = 0; x < MAX_IM; x++)
        {
            model[x] = 0.;
        }

        for (y = 0; y < nrows; y++)
        {
            for (x = 0; x < ncols; x++)
            {
                for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++)
                {
                    xx = zeta[zeta_index(x, y, m)].x;
                    iy = zeta[zeta_index(x, y, m)].iy;
                    ww = zeta[zeta_index(x, y, m)].w;
                    model[im_index(x, y)] += sP[xx] * sL[iy] * ww;
                }
            }
        }

        /* Compare model and data */
        // We use the Median absolute derivation to estimate the distribution
        // The MAD is more robust than the usual STD as it uses the median
        // However the MAD << STD, since we are not dealing with a Gaussian
        // at all, but a distribution with heavy wings.
        // Therefore we use the factor 40, instead of 6 to estimate a reasonable range
        // of values. The cutoff is roughly the same.
        // Technically the distribution might best be described by a Voigt profile
        // which we then would have to fit to the distrubtion and then determine,
        // the range that covers 99% of the data.
        // Since that is much more complicated we just use the MAD.
        /* Compute sigma for outlier rejection (RMS of residuals) */
        cost = 0;
        tmp = 0;
        isum = 0;
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                if (mask[im_index(x, y)])
                {
                    double resid = model[im_index(x, y)] - im[im_index(x, y)];
                    tmp += resid * resid;
                    double resid_scaled = resid / max(pix_unc[im_index(x, y)], 1);
                    cost += resid_scaled * resid_scaled;
                    isum++;
                }
            }
        }
        cost /= (isum - (ncols + ny));
        dev = sqrt(tmp / isum);

        /* Adjust the mask marking outliers */
        if (kappa > 0)
        {
            for (y = 0; y < nrows; y++)
            {
                for (x = delta_x; x < ncols - delta_x; x++)
                {
                    if (fabs(model[im_index(x, y)] - im[im_index(x, y)]) < kappa * dev)
                        mask[im_index(x, y)] = 1;
                    else
                        mask[im_index(x, y)] = 0;
                }
            }
        }

#if DEBUG
        if (cost == 0)
        {
            printf("Iteration: %i, Reduced chi-square: %f\n", iter, cost);
            printf("dev: %f\n", dev);
            printf("isum: %i\n", isum);
            printf("iteration: %i\n", iter);
            printf("-----------\n");
        }
#endif
        /* Check for convergence. maxiter is an unconditional upper bound;
           previously the non-finite-cost retry bypassed it and could hang
           forever when the solver produced NaNs (e.g. kappa=0 leaves NaN
           cells un-masked). */
    } while ((iter++ < maxiter) && ((cost_old - cost > ftol) || !isfinite(cost) || !isfinite(cost_old)));

    if (iter >= maxiter - 1)
    {
        status = -1; // ran out of iterations
        success = 0;
    }
    else if (cost_old - cost <= ftol)
        status = 1; // cost did not improve enough between iterations

    /* Uncertainty estimate */

    for (x = 0; x < ncols_fine; x++)
    {
        unc[sp_index(x)] = 0.;
        p_bj[pbj_index(x)] = 0.;
        p_Aij[paij_index(x, 0)] = 0;
    }

    for (y = 0; y < nrows; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++) // Loop through all pixels contributing to x,y
            {
                if (mask[im_index(x, y)])
                {
                    // Should pix_unc contribute here?
                    xx = zeta[zeta_index(x, y, m)].x;
                    iy = zeta[zeta_index(x, y, m)].iy;
                    ww = zeta[zeta_index(x, y, m)].w;
                    tmp = im[im_index(x, y)] - model[im_index(x, y)];
                    unc[sp_index(xx)] += tmp * tmp * ww;
                    p_bj[pbj_index(xx)] += ww;           // Norm
                    p_Aij[paij_index(xx, 0)] += ww * ww; // Norm squared
                }
            }
        }
    }

    for (x = 0; x < ncols_fine; x++)
    {
        norm = p_bj[pbj_index(x)] - p_Aij[paij_index(x, 0)] / p_bj[pbj_index(x)];
        unc[sp_index(x)] = sqrt(unc[sp_index(x)] / norm * nrows);
    }

    for (x = 0; x < delta_x * osamp_spec; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }
    for (x = ncols_fine - delta_x * osamp_spec; x < ncols_fine; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }

    free(l_Aij);
    free(p_Aij);
    free(p_bj);
    free(l_bj);

    free(xi);
    free(zeta);
    free(m_zeta);

    info[0] = success;
    info[1] = cost;
    info[2] = status;
    info[3] = iter;
    info[4] = delta_x;

    return 0;
}

int create_spectral_model(int ncols, int nrows, int osample, xi_ref* xi, double* spec, double* slitfunc, double* img){
    int ny, pix_x, pix_y, x, iy, m;
    double pix_w;

    ny = (nrows + 1) * osample + 1;

    for (x = 0; x < ncols; x++)
    {
        for (iy = 0; iy < nrows+1; iy++)
        {
            img[im_index(x, iy)] = 0;
        }

    }

    for (x = 0; x < ncols; x++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (m = 0; m < 4; m++)
            {
                pix_x = xi[xi_index(x, iy, m)].x;
                pix_y = xi[xi_index(x, iy, m)].y;
                pix_w = xi[xi_index(x, iy, m)].w;
                if ((pix_x != -1) && (pix_y != -1) && (pix_w != 0)){
                    img[im_index(pix_x, pix_y)] += pix_w * spec[x] * slitfunc[iy];
                }
            }
        }
    }
    return 0;
}
