# Notes on Python to C interface


In the following copy of the slitdec function interface from slitdec.c
there are comments that describe if each argument should be exposed in the 
nanobind wrapper, or calculated by it, and how. some should have default values in python.
output arrays are also marked.

int slitdec(        int ncols, // X-dimension of im
                    int nrows, // Y-dimension of im
                    double *im, // exposed, main input image
                    double *pix_unc, // exposed, pixel error map
                    unsigned char *mask, // exposed, pixel mask
                    double *ycen, // exposed, array with length ncols
                    double *slitcurve, // exposed, polynomial coefficients
                    double *slitdeltas, // exposed, array length nrows
                    int osample, // exposed, default 6
                    double lambda_sP, // exposed, default 0.0
                    double lambda_sL, // exposed, default 0.1
                    int maxiter, // exposed, default 20
                    double *sP,  // output spectrum
                    double *sL,  // output slit-function
                    double *model, // output model 
                    double *unc,  // output, spectrum errors
                    double *info  // output, status info
                    )
