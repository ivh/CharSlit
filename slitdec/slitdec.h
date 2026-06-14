typedef struct
{
    int x;
    int iy;   /* Contributing subpixel  x,iy      */
    double w; /* Contribution weight <= 1/osample */
} zeta_ref;

/* Key ranges of one pixel's zeta list, maintained by the zeta build.
   The SLE fill loops merge the list by subpixel index iy (sL system) or
   source column x (sP system) into a small dense window [min, max]
   instead of searching a list of unique keys. */
typedef struct
{
    int min_iy, max_iy;
    int min_x, max_x;
} zeta_rng;

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
                    double *info);

int zeta_tensors(
    int ncols,
    int nrows,
    int ny,
    double *ycen,
    int *ycen_offset,
    int y_lower_lim,
    int osample,
    int osamp_spec,
    double *slitcurve,
    double *slitdeltas,
    zeta_ref *zeta,
    int *m_zeta,
    zeta_rng *z_rng);
