/*
 * IDL CALL_EXTERNAL interface for slitdec_ox (REDUCE).
 *
 * Entry point: slit_func_ox. Every parameter is an argument: there are no
 * defaults and nothing is read from the environment, so a call shows exactly
 * which values were used. All 27 arguments are required; IDL long = C int.
 *
 *   0  ncols       long   scalar            swath width in pixels
 *   1  nrows       long   scalar            swath height in pixels
 *   2  osample     long   scalar            subpixels per pixel of the slit function (y)
 *   3  osample_x   long   scalar            fine spectrum bins per pixel (x), >= 1
 *   4  im          double [ncols,nrows]     swath image
 *   5  ycen_frac   double [ncols]           fractional trace position, 0 <= v < 1
 *   6  ycen_offset long   [ncols]           integer part of the trace position
 *   7  y_lower_lim long   scalar            rows below the trace row in the swath
 *   8  PSF_curve   double [3,ncols]         slit curvature: shift = c1*dy + c2*dy^2
 *                                           (element 0 unused)
 *   9  mu1_x       double scalar            spectrum penalties on the 1st, 2nd and 3rd
 *  10  mu2_x       double scalar              derivative, in physical units: the
 *  11  mu3_x       double scalar              spectrum is filtered by
 *                                             1/(1 + sum_k mu_k nu^2k), nu in rad/pixel
 *  12  mu1_y       double scalar            slit-function penalties on the 1st and 2nd
 *  13  mu2_y       double scalar              derivative, same units; at least one > 0
 *  14  maxiter     long   scalar            maximum number of iterations, >= 1
 *  15  kappa       double scalar            outlier rejection threshold in sigma
 *                                             (<= 0: no rejection)
 *  16  weight_mode long   scalar            0: unweighted least squares
 *                                           1: optimal-extraction weights 1/variance with
 *                                              variance = model/gain + (read_noise/gain)^2
 *                                              [ADU^2], from the current model (not the
 *                                              data) at every iteration; the first
 *                                              iteration is unweighted
 *  17  gain        double scalar            electrons per ADU (> 0; used when weighted)
 *  18  read_noise  double scalar            read noise in electrons (> 0 when weighted:
 *                                              it also keeps the weights finite where
 *                                              the model is 0)
 *  19  sL_preset   long   scalar            non-zero: sL is given and held fixed
 *  20  mask        byte   [ncols,nrows]     in/out, 0 = bad (updated by the kappa clip)
 *  21  model       double [ncols,nrows]     OUTPUT model swath
 *  22  sP          double [ncols]           OUTPUT spectrum per detector pixel
 *  23  sP_fine     double [ncols*osample_x] OUTPUT fine spectrum
 *  24  sL          double [ny]              in/out slit function,
 *                                             ny = (nrows+1)*osample + 1
 *  25  unc         double [ncols]           OUTPUT spectrum uncertainty
 *  26  info        double [11]              OUTPUT: 0 success, 1 last sP change,
 *                                             2 status, 3 iterations, 4 delta_x (columns
 *                                             at each swath edge that are set to 0),
 *                                             5 dropped zeta entries, 6 sP band width,
 *                                             7 damping ratio x, 8 resolution x [px],
 *                                             9 damping ratio y, 10 resolution y [px]
 *                                             (damping < 1: the penalty can ring)
 *
 * Returns 0 on success, -1 on a wrong argument (a message is printed), otherwise
 * the return value of slitdec_ox2.
 *
 * Memory layout: IDL [ncols,nrows] is column-major, i.e. C [nrows][ncols], so the
 * flat index y*ncols + x is the same on both sides and no copy is needed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "slitdec_ox.h"

#define NARGS 27
#define NINFO 11

#ifdef BUILDING_REDUCE_WIN_DLL
#define REDUCE_DLL __declspec(dllexport)
#else
#define REDUCE_DLL
#endif

static int bad(const char *what)
{
    printf("slit_func_ox: %s\n", what);
    return -1;
}

int REDUCE_DLL slit_func_ox(int argc, void *argv[])
{
    int ncols, nrows, osample, osample_x, y_lower_lim, maxiter, sL_preset, weight_mode;
    double mu_x[3], mu_y[2], kappa, gain, read_noise;
    double *im, *ycen_frac, *PSF_curve, *model, *sP, *sP_fine, *sL, *unc, *info_out;
    int *ycen_offset;
    unsigned char *mask;
    int x, y, k, ny, ret;
    double info[NINFO] = {0};

    if (argc != NARGS)
    {
        printf("slit_func_ox: requires exactly %d arguments, got %d:\n", NARGS, argc);
        printf("  ncols, nrows, osample, osample_x, im, ycen_frac, ycen_offset, y_lower_lim,\n");
        printf("  PSF_curve, mu1_x, mu2_x, mu3_x, mu1_y, mu2_y, maxiter, kappa,\n");
        printf("  weight_mode, gain, read_noise, sL_preset,\n");
        printf("  mask, model, sP, sP_fine, sL, unc, info\n");
        return -1;
    }

    ncols       = *(int *)argv[0];
    nrows       = *(int *)argv[1];
    osample     = *(int *)argv[2];
    osample_x   = *(int *)argv[3];
    im          = (double *)argv[4];
    ycen_frac   = (double *)argv[5];
    ycen_offset = (int *)argv[6];
    y_lower_lim = *(int *)argv[7];
    PSF_curve   = (double *)argv[8];
    for (k = 0; k < 3; k++)
        mu_x[k] = *(double *)argv[9 + k];
    for (k = 0; k < 2; k++)
        mu_y[k] = *(double *)argv[12 + k];
    maxiter     = *(int *)argv[14];
    kappa       = *(double *)argv[15];
    weight_mode = *(int *)argv[16];
    gain        = *(double *)argv[17];
    read_noise  = *(double *)argv[18];
    sL_preset   = *(int *)argv[19];
    mask        = (unsigned char *)argv[20];
    model       = (double *)argv[21];
    sP          = (double *)argv[22];
    sP_fine     = (double *)argv[23];
    sL          = (double *)argv[24];
    unc         = (double *)argv[25];
    info_out    = (double *)argv[26];

    if (ncols < 1 || nrows < 1)
        return bad("ncols and nrows must be >= 1");
    if (osample < 1 || osample_x < 1)
        return bad("osample and osample_x must be >= 1");
    if (y_lower_lim < 0 || y_lower_lim >= nrows)
        return bad("y_lower_lim must be in 0 .. nrows-1");
    for (k = 0; k < 3; k++)
        if (!(mu_x[k] >= 0.0) || !isfinite(mu_x[k]))
            return bad("mu1_x, mu2_x, mu3_x must be finite and >= 0");
    for (k = 0; k < 2; k++)
        if (!(mu_y[k] >= 0.0) || !isfinite(mu_y[k]))
            return bad("mu1_y, mu2_y must be finite and >= 0");
    if (mu_y[0] == 0.0 && mu_y[1] == 0.0)
        return bad("mu1_y or mu2_y must be > 0: without a penalty the slit function is ill-posed");
    if (maxiter < 1)
        return bad("maxiter must be >= 1");
    if (!isfinite(kappa))
        return bad("kappa must be finite");
    if (weight_mode != 0 && weight_mode != 1)
        return bad("weight_mode must be 0 (none) or 1 (model variance)");
    if (weight_mode == 1 && !(gain > 0.0 && isfinite(gain)))
        return bad("gain must be > 0 with weight_mode 1");
    if (weight_mode == 1 && !(read_noise > 0.0 && isfinite(read_noise)))
        return bad("read_noise must be > 0 (electrons) with weight_mode 1");

    ny = osample * (nrows + 1) + 1;

    /* slitdec_ox takes the full trace position and splits it itself;
       it overwrites this copy with the fractional part.                    */
    double *ycen = malloc(ncols * sizeof(double));
    double *slitcurve = calloc((size_t)ncols * 6, sizeof(double));
    double *slitdeltas = calloc(ny, sizeof(double));
    double *pix_unc = calloc((size_t)nrows * ncols, sizeof(double)); /* unused here */
    if (!ycen || !slitcurve || !slitdeltas || !pix_unc)
    {
        free(ycen); free(slitcurve); free(slitdeltas); free(pix_unc);
        return bad("out of memory");
    }
    for (x = 0; x < ncols; x++)
    {
        ycen[x] = ycen_frac[x] + ycen_offset[x];
        /* IDL PSF_curve[3,ncols]: element [k,x] at k + 3*x. slitdec uses
           slitcurve[x*6 + k]; the constant term k = 0 is not used.         */
        slitcurve[x * 6 + 1] = PSF_curve[1 + 3 * x];
        slitcurve[x * 6 + 2] = PSF_curve[2 + 3 * x];
    }

    /* Initial spectrum from the image, as the original slit_func_2d does */
    for (x = 0; x < ncols; x++)
    {
        double norm = 0.0;
        sP[x] = 0.0;
        for (y = 0; y < nrows; y++)
        {
            norm += mask[y * ncols + x];
            sP[x] += fabs(im[y * ncols + x] * mask[y * ncols + x]);
        }
        norm /= nrows;
        sP[x] = norm > 0.0 ? sP[x] / norm : -1000.0;
    }

    /* slitdec_ox2 weight_mode 2 uses w = 1/(model + noise_floor). With the model in ADU,
       variance = model/gain + (read_noise/gain)^2 = (model + read_noise^2/gain)/gain;
       the weights are normalised, so the common factor 1/gain drops out.           */
    info[4] = -11; /* ask for the extended info */
    ret = slitdec_ox2(ncols, nrows, im, pix_unc, mask, ycen, slitcurve, slitdeltas,
                      osample, osample_x, mu_x, 3, mu_y, 2,
                      0.0,       /* lambda_sL: not used, mu_y is the only sL penalty */
                      maxiter, kappa, sL_preset,
                      weight_mode == 1 ? 2 : 0,
                      weight_mode == 1 ? read_noise * read_noise / gain : -1.0,
                      sP, sP_fine, sL, model, unc, info, y_lower_lim);

    for (k = 0; k < NINFO; k++)
        info_out[k] = info[k];

    free(ycen);
    free(slitcurve);
    free(slitdeltas);
    free(pix_unc);
    return ret;
}
