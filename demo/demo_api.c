/* Thin shim around slitdec() for the wasm demo: builds the constant-ycen,
   uniform-slitcurve inputs so the JS side only handles image + scalars. */

#include <stdlib.h>
#include <math.h>

#include "../slitdec/slitdec.h"

int demo_extract(int ncols, int nrows, double *im,
                 double ycen_frac, double c1, double c2,
                 int osample, double lambda_sP, double lambda_sL,
                 int maxiter, double kappa,
                 double *sP, double *sL, double *model,
                 double *unc, double *info, unsigned char *mask)
{
    int ny = osample * (nrows + 1) + 1;
    int npix = ncols * nrows;

    double *ycen = malloc(ncols * sizeof(double));
    double *slitcurve = calloc((size_t)ncols * 6, sizeof(double));
    double *slitdeltas = calloc(ny, sizeof(double));
    /* unused by the algorithm but must be a valid array */
    double *pix_unc = malloc((size_t)npix * sizeof(double));

    if (!ycen || !slitcurve || !slitdeltas || !pix_unc)
    {
        free(ycen);
        free(slitcurve);
        free(slitdeltas);
        free(pix_unc);
        info[0] = 0;
        info[2] = -99;
        return -99;
    }

    for (int x = 0; x < ncols; x++)
    {
        ycen[x] = ycen_frac;
        slitcurve[x * 6 + 1] = c1;
        slitcurve[x * 6 + 2] = c2;
    }

    /* mask is recomputed each run because slitdec modifies it in place */
    for (int i = 0; i < npix; i++)
    {
        mask[i] = isfinite(im[i]) ? 1 : 0;
        pix_unc[i] = 1.0;
    }

    for (int i = 0; i < ncols; i++)
    {
        sP[i] = 1.0;
        unc[i] = 0.0;
    }
    for (int i = 0; i < ny; i++)
        sL[i] = 1.0 / osample;
    for (int i = 0; i < npix; i++)
        model[i] = 0.0;

    int ret = slitdec(ncols, nrows, im, pix_unc, mask, ycen,
                      slitcurve, slitdeltas,
                      NULL, 0 /* no slit kernel */,
                      osample, lambda_sP, lambda_sL, maxiter, kappa,
                      0 /* use_preset */, 0 /* unweighted */, -1.0 /* noise floor */,
                      sP, sL, model, unc, info);

    free(ycen);
    free(slitcurve);
    free(slitdeltas);
    free(pix_unc);
    return ret;
}
