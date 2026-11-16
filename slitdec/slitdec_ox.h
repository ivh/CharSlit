/* slitdec with oversampling of the spectrum along x (see slitdec_ox.c). */

typedef struct
{
    int x;    /* Contributing fine spectrum bin (x * osample_x + p) */
    int iy;   /* Contributing slit-function subpixel                */
    double w; /* Contribution weight                                */
} zeta_ref_ox;

typedef struct
{
    int min_iy, max_iy;
    int min_x, max_x;   /* in fine-bin units */
} zeta_rng_ox;

#define SLITDEC_OX_KMAX 15 /* highest derivative order of the penalties */

/* Penalties sum_k mu[k-1] * F * int (d^k s)^2 in physical units for the spectrum
   (mu_x, nk_x orders) and the slit function (mu_y, nk_y orders; none: slitdec's
   lambda_sL). lambda_sP: slitdec's unscaled first-difference penalty, applied
   between fine bins. t_offset: where along a y-subpixel the curvature is
   evaluated (0.5 centre, 0 slitdec.c). y_lower_lim: rows below the trace row in
   the swath, < 0 means nrows / 2. unc_mode 0: residual-based unc; 1: propagated
   pixel variances; 2: propagated, scaled by chi^2; cov (may be NULL):
   (ncols*out_x) x (nlag+1) covariances on the out_x grid. See slitdec_ox.c. */
int slitdec_ox2(int ncols, int nrows, double *im, double *pix_unc,
                unsigned char *mask, double *ycen, double *slitcurve,
                double *slitdeltas, int osample, int osample_x,
                const double *mu_x, int nk_x, const double *mu_y, int nk_y,
                double lambda_sP, double lambda_sL, int maxiter, double kappa,
                int use_preset, int weight_mode, double noise_floor,
                double t_offset,
                double *sP, double *sP_fine, double *sL, double *model,
                double *unc, double *info, int y_lower_lim,
                int unc_mode, int out_x, int nlag, double *cov, double *chi2);
