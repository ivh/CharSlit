#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include "slitdec.h"

/* ---------- Minimal parallel-for ----------
   Work is split into a FIXED number of chunks (PAR_CHUNKS, independent of
   the thread count); threads grab chunks with an atomic counter and each
   chunk accumulates into its own partial buffers, which the caller then
   reduces in chunk order. Results are therefore identical for any number
   of threads >= 2; nthreads == 1 (or small images) takes the plain
   sequential code path. Thread count: CHARSLIT_THREADS env var, default
   min(8, online cores). */
#define PAR_CHUNKS 32
#define PAR_MAX_THREADS 64

/* Persistent worker pool: created once per slitdec call, woken per
   parallel region (the fills run ~5 regions per iteration; spawning
   threads each time measurably added up). pool == NULL degrades to
   running the chunks sequentially on the caller. */
typedef struct
{
    pthread_t th[PAR_MAX_THREADS];
    int nworkers; /* excluding the calling thread */
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    int generation;
    int shutdown;
    int active;
    void (*fn)(void *, int);
    void *ctx;
    _Atomic int next;
    int nchunks;
} pool_t;

static void pool_work(pool_t *p)
{
    int k;
    while ((k = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed)) < p->nchunks)
        p->fn(p->ctx, k);
}

static void *pool_thread(void *arg)
{
    pool_t *p = arg;
    int mygen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;)
    {
        while (p->generation == mygen && !p->shutdown)
            pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->shutdown)
            break;
        mygen = p->generation;
        pthread_mutex_unlock(&p->mu);
        pool_work(p);
        pthread_mutex_lock(&p->mu);
        if (--p->active == 0)
            pthread_cond_signal(&p->cv_done);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

static pool_t *pool_start(int nthreads)
{
    if (nthreads <= 1)
        return NULL;
    if (nthreads > PAR_MAX_THREADS)
        nthreads = PAR_MAX_THREADS;
    pool_t *p = calloc(1, sizeof(pool_t));
    if (!p)
        return NULL;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    for (int i = 0; i < nthreads - 1; i++)
    {
        if (pthread_create(&p->th[i], NULL, pool_thread, p) != 0)
            break;
        p->nworkers++;
    }
    return p;
}

static void pool_run(pool_t *p, void (*fn)(void *, int), void *ctx, int nchunks)
{
    if (!p || p->nworkers == 0)
    {
        for (int k = 0; k < nchunks; k++)
            fn(ctx, k);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    p->nchunks = nchunks;
    p->active = p->nworkers;
    p->generation++;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    pool_work(p); /* the caller participates */
    pthread_mutex_lock(&p->mu);
    while (p->active > 0)
        pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

static void pool_stop(pool_t *p)
{
    if (!p)
        return;
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nworkers; i++)
        pthread_join(p->th[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    free(p);
}

static int par_nthreads(int ncols, int nrows)
{
    const char *env = getenv("CHARSLIT_THREADS");
    long n;
    if (env && *env)
        n = atol(env);
    else
    {
        n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > 12)
            n = 12;
    }
    if (n < 1)
        n = 1;
    /* Small images: thread spawn overhead exceeds the work */
    if ((size_t)ncols * nrows < 100000)
        n = 1;
    return (int)n;
}

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
int _nrows = 0;
int _ny = 0;
int _nx = 0;
int _osample = 0;
int _n = 0;
int _nd = 0;
#else
#define _ncols ncols
#define _nrows nrows
#define _ny ny
#define _nx nx
#define _osample osample
#define _n n
#define _nd nd
#endif

// Define the sizes of each array
#define MAX_ZETA_X (_ncols)
#define MAX_ZETA_Y (_nrows)
#define MAX_ZETA_Z (3 * ((_osample) + 1))
#define MAX_ZETA (MAX_ZETA_X * MAX_ZETA_Y * MAX_ZETA_Z)
#define MAX_MZETA ((_ncols) * (_nrows))
#define MAX_CRV_X (_ncols)
#define MAX_CRV_Y (3)
#define MAX_CRV (MAX_CRV_X * MAX_CRV_Y)
#define MAX_A ((_n) * (_nd))
#define MAX_R (_n)
#define MAX_SP (_ncols)
#define MAX_SL (_ny)
#define MAX_LAIJ_X (_ny)
#define MAX_LAIJ_Y (4 * (_osample) + 1)
#define MAX_LAIJ (MAX_LAIJ_X * MAX_LAIJ_Y)
#define MAX_PAIJ_X (_ncols)
#define MAX_PAIJ_Y (_nx)
#define MAX_PAIJ (MAX_PAIJ_X * MAX_PAIJ_Y)
#define MAX_LBJ (_ny)
#define MAX_PBJ (_ncols)
#define MAX_IM ((_ncols) * (_nrows))

// If we want to check the index use functions to represent the index
// Otherwise a simpler define will do, which should be faster ?
#if DEBUG
static long zeta_index(long x, long y, long z)
{
    long i = z + x * MAX_ZETA_Z + y * MAX_ZETA_Z * _ncols;
    if ((i < 0) | (i >= MAX_ZETA))
    {
        printf("INDEX OUT OF BOUNDS. Zeta[%li, %li, %li]\n", x, y, z);
        return 0;
    }
    return i;
}

static long mzeta_index(long x, long y)
{
    long i = x + y * _ncols;
    if ((i < 0) | (i >= MAX_MZETA))
    {
        printf("INDEX OUT OF BOUNDS. Mzeta[%li, %li]\n", x, y);
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
    long i = _nd * x + y;
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
    long i = (x)*MAX_LAIJ_Y + (y);
    if ((i < 0) | (i >= MAX_LAIJ))
    {
        printf("INDEX OUT OF BOUNDS. l_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long paij_index(long x, long y)
{
    long i = (x)*_nx + (y);
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
/* zeta is y-major: one detector row's lists are contiguous, so the fills,
   the model and the uncertainty pass can iterate rows outermost (which keeps
   their band-matrix windows cache-resident) while still reading zeta
   sequentially. */
#define zeta_index(x, y, z) ((z) + (x)*MAX_ZETA_Z + (y)*MAX_ZETA_Z * _ncols)
#define mzeta_index(x, y) ((x) + (y)*_ncols)
#define curve_index(x, y) ((x)*6 + (y))
// Band matrices are stored row-major (band entries for one row are
// contiguous): this matches the access pattern of both the SLE fill
// loops and bandsol, unlike the previous column-major layout.
#define a_index(x, y) ((x)*nd + (y))
#define r_index(i) (i)
#define sp_index(i) (i)
#define laij_index(x, y) ((x) * (4 * osample + 1) + (y))
#define paij_index(x, y) ((x)*nx + (y))
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

// Fast median/percentile via quickselect.
// Algorithm from Numerical recipes in C of 1992
// see http://ndevilla.free.fr/median/median/
#define ELEM_SWAP(a, b)          \
    {                            \
        register double t = (a); \
        (a) = (b);               \
        (b) = t;                 \
    }

double quick_select_median(double arr[], unsigned int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;)
    {
        if (high <= low) /* One element only */
            return arr[median];
        if (high == low + 1)
        { /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }
        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);
        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);
        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;)
        {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);
            if (hh < ll)
                break;
            ELEM_SWAP(arr[ll], arr[hh]);
        }
        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);
        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

/* Quickselect for arbitrary percentile (0-100) */
double quick_select_percentile(double arr[], unsigned int n, double percentile)
{
    int low, high;
    int target;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    target = (int)((percentile / 100.0) * (n - 1));
    for (;;)
    {
        if (high <= low)
            return arr[target];
        if (high == low + 1)
        {
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[target];
        }
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);
        ELEM_SWAP(arr[middle], arr[low + 1]);
        ll = low + 1;
        hh = high;
        for (;;)
        {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);
            if (hh < ll)
                break;
            ELEM_SWAP(arr[ll], arr[hh]);
        }
        ELEM_SWAP(arr[low], arr[hh]);
        if (hh <= target)
            low = ll;
        if (hh >= target)
            high = hh - 1;
    }
}

static inline void zeta_add(zeta_ref *zeta, int *m_zeta, zeta_rng *z_rng,
                     int ncols, int nrows, int osample,
                     int x, int iy, int xx, int yy, double w)
{
    if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
    {
        const int m = m_zeta[mzeta_index(xx, yy)];
        /* Extreme geometry can feed one pixel from more subpixels than the
           fixed per-pixel list holds; drop the entry rather than overflow
           into the next pixel's list. Real data reaches ~18 of 21 slots. */
        if (m >= MAX_ZETA_Z)
            return;
        zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
        zeta[zeta_index(xx, yy, m)].x = x;
        zeta[zeta_index(xx, yy, m)].iy = iy;
        zeta[zeta_index(xx, yy, m)].w = w;
        m_zeta[mzeta_index(xx, yy)]++;
        if (iy < zr->min_iy) zr->min_iy = iy;
        if (iy > zr->max_iy) zr->max_iy = iy;
        if (x < zr->min_x) zr->min_x = x;
        if (x > zr->max_x) zr->max_x = x;
    }
}

static void zeta_build_cols(
    int ncols, int nrows, int xsrc0, int xsrc1, int xown0, int xown1,
    double *ycen, int *ycen_offset, int osample,
    double *slitcurve, double *slitdeltas,
    int *iy1c, int *iy2c, double *d1c, double *d2c, double *dyc,
    zeta_ref *zeta, int *m_zeta, zeta_rng *z_rng);

int zeta_tensors(
    int ncols,
    int nrows,
    int ny,
    double *ycen,
    int *ycen_offset,
    int y_lower_lim,
    int osample,
    double *slitcurve,
    double *slitdeltas,
    zeta_ref *zeta,
    int *m_zeta,
    zeta_rng *z_rng)
{
    /*
    Create the zeta tensor, which describes the contribution of each subpixel
    of the oversampled slit function to each detector pixel, considering the
    curvature of the slit.

    Historically this routine also built the inverse mapping ("xi" tensor,
    subpixel -> detector pixels). Since the SLE fill loops in slitdec became
    pixel-centric, only zeta is needed, which also collapses the bookkeeping
    of which xi corner (LL/LR/UL/UR) a contribution belongs to: the zeta
    insertions were identical for all corner cases.

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
    slitcurve : double array of shape (ncols, 6)
        Polynomial fit to the slit image curvature.
        For column d_x = sum_k slitcurve[ncols][k] * d_y^k,
        where d_y is the offset from the central line ycen.
    slitdeltas : double array of shape (ny,)
        Additional per-subpixel horizontal offsets
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
    int x, xx, y, yy, ix1, ix2, iy, iy1, iy2;
    double step, delta, dy, w, d1, d2;

    step = 1.e0 / osample;

    /* Clean zeta counts. The zeta entries themselves need no initialization:
       only the first m_zeta[x, y] entries of each list are ever read.
       Same for the key ranges: only read where m_zeta[x, y] > 0. */
    for (x = 0; x < ncols; x++)
        for (y = 0; y < nrows; y++)
        {
            zeta_rng *zr = &z_rng[mzeta_index(x, y)];
            m_zeta[mzeta_index(x, y)] = 0;
            zr->min_iy = INT_MAX;
            zr->max_iy = INT_MIN;
            zr->min_x = INT_MAX;
            zr->max_x = INT_MIN;
        }

    /*
    Construct the zeta tensor. It contains pixel references and contribution
    values coming from subpixels to a given detector pixel.
    Note that zeta is used in the equations for sL, sP and for the model but it
    does not involve the data, only the geometry. Thus it can be pre-computed once.
    */
    /* The loop is row-outer so that zeta (y-major) is written near-
       sequentially. The per-column recurrences (iy1, iy2, dy) live in
       small state arrays and execute exactly the same operation sequence
       per column as the historic column-outer loop, so all inserted
       values are bit-identical; only the order of entries within one
       pixel's list changes (sums over them reorder at the rounding
       level). */
    int *iy1c = malloc(ncols * sizeof(int));
    int *iy2c = malloc(ncols * sizeof(int));
    double *d1c = malloc(ncols * sizeof(double));
    double *d2c = malloc(ncols * sizeof(double));
    double *dyc = malloc(ncols * sizeof(double));

    (void)ny;
    for (x = 0; x < ncols; x++)
    {
        /*
        I promised to reconsider the initial offset. Here it is. For the original layout
        (no column shifts and discontinuities in ycen) there is pixel y that contains the
        central line yc. There are two options here (by construction of ycen that can be 0
        but cannot be 1): (1) yc is inside pixel y and (2) yc falls at the boundary between
        pixels y and y-1. yc cannot be at the boundary of pixels y+1 and y because we would
        select y+1 to be pixel y in that case.

        Next we need to define starting and ending indices iy for sL subpixels that contribute
        to pixel y. I call them iy1 and iy2. For both cases we assume osample+1 subpixels covering
        pixel y (wierd). So for case 1 iy1 will be (y-1)*osample and iy2 == y*osample. Special
        treatment of the boundary subpixels will compensate for introducing extra subpixel in
        case 1. In case 2 things are more logical: iy1=(yc-y)*osample+(y-1)*osample;
        iy2=(y+1-yc)*osample)+(y-1)*osample. ycen is yc-y making things simpler. Note also that
        the same pattern repeates for all rows: we only need to initialize iy1 and iy2 and keep
        incrementing them by osample.
        */
        iy2c[x] = osample - floor(ycen[x] * osample);
        iy1c[x] = iy2c[x] - osample;

        /*
        Handling partial subpixels cut by detector pixel rows is again tricky. Here we have three
        cases (mostly because of the decision to assume that we always have osample+1 subpixels
        per one detector pixel). Here d1 is the fraction of the subpixel iy1 inside detector pixel y.
        d2 is then the fraction of subpixel iy2 inside detector pixel y. By definition d1+d2==step.
        Case 1: ycen falls on the top boundary of each detector pixel (ycen == 1). Here we conclude
                that the first subpixel is fully contained inside pixel y and d1 is set to step.
        Case 2: ycen falls on the bottom boundary of each detector pixel (ycen == 0). Here we conclude
                that the first subpixel is totally outside of pixel y and d1 is set to 0.
        Case 3: ycen falls inside of each pixel (0>ycen>1). In this case d1 is set to the fraction of
                the first step contained inside of each pixel.
        And BTW, this also means that central line coinsides with the upper boundary of subpixel iy2
        when the y loop reaches pixel y_lower_lim. In other words:

        dy=(iy-(y_lower_lim+ycen[x])*osample)*step-0.5*step
        */
        d1c[x] = fmod(ycen[x], step);
        if (d1c[x] == 0)
            d1c[x] = step;
        d2c[x] = step - d1c[x];

        /* Define initial distance from ycen       */
        /* It is given by the center of the first  */
        /* subpixel falling into pixel y_lower_lim */
        dyc[x] = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;
    }

    /*
    Now we go detector pixels x and y incrementing subpixels looking for their contributions
    to the current and adjacent pixels. Note that the curvature/tilt of the projected slit
    image could be so large that subpixel iy may not contribute to column x at all. On the
    other hand, subpixels around ycen by definition must contribute to pixel x,y.

    Each subpixel is assumed to be exactly 1 detector pixel wide; a horizontal shift delta
    divides its weight w between columns ix1=int(delta) and ix2=ix1+signum(delta) as
    (1-|delta-ix1|)*w and |delta-ix1|*w. The yy offset is required because the iy subpixel
    contributes to the yy row in the xx column of detector pixels where yy and y are in the
    same row. In the packed array this is not necessarily true. Instead, what we know is:
    y+ycen_offset[x] == yy+ycen_offset[xx]
    */
    zeta_build_cols(ncols, nrows, 0, ncols, 0, ncols, ycen, ycen_offset,
                    osample, slitcurve, slitdeltas, iy1c, iy2c, d1c, d2c, dyc,
                    zeta, m_zeta, z_rng);

    free(iy1c);
    free(iy2c);
    free(d1c);
    free(d2c);
    free(dyc);
    return 0;
}

/* Process source columns [xsrc0, xsrc1) but insert only entries whose
   target column falls in [xown0, xown1): a source column feeds target
   columns at most the maximum horizontal shift away, so parallel chunks
   re-process a small halo of source columns around their owned range and
   each pixel's entries are then inserted by exactly one chunk, in the same
   (y, x, iy) order as the sequential build - the zeta content is
   bit-identical for any chunking. The per-column recurrence state
   (iy1c/iy2c/dyc) runs complete within one chunk, so no state hand-off
   between chunks is needed. */
static void zeta_build_cols(
    int ncols, int nrows, int xsrc0, int xsrc1, int xown0, int xown1,
    double *ycen, int *ycen_offset, int osample,
    double *slitcurve, double *slitdeltas,
    int *iy1c, int *iy2c, double *d1c, double *d2c, double *dyc,
    zeta_ref *zeta, int *m_zeta, zeta_rng *z_rng)
{
    int x, xx, y, yy, ix1, ix2, iy, iy1, iy2;
    double step, delta, w;
    step = 1.e0 / osample;

    for (y = 0; y < nrows; y++)
    {
        for (x = xsrc0; x < xsrc1; x++)
        {
            const double d1 = d1c[x], d2 = d2c[x];
            double dy = dyc[x];
            iy1 = iy1c[x] += osample; // Bottom subpixel falling in row y
            iy2 = iy2c[x] += osample; // Top subpixel falling in row y
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
                double t = dy - ycen[x];
                delta = t * (slitcurve[curve_index(x, 1)] +
                        t * (slitcurve[curve_index(x, 2)] +
                        t * (slitcurve[curve_index(x, 3)] +
                        t * (slitcurve[curve_index(x, 4)] +
                        t *  slitcurve[curve_index(x, 5)]))))
                        + slitdeltas[iy];
                ix1 = delta;
                ix2 = ix1 + signum(delta);

                if (ix1 < ix2) /* Subpixel iy shifts to the right from column x */
                {
                    if (x + ix1 >= 0 && x + ix2 < ncols)
                    {
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        if (xx >= xown0 && xx < xown1)
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                     w - fabs(delta - ix1) * w);
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        if (xx >= xown0 && xx < xown1)
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                     fabs(delta - ix1) * w);
                    }
                }
                else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                {
                    if (x + ix2 >= 0 && x + ix1 < ncols)
                    {
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        if (xx >= xown0 && xx < xown1)
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                     fabs(delta - ix1) * w);
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        if (xx >= xown0 && xx < xown1)
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                     w - fabs(delta - ix1) * w);
                    }
                }
                else /* Subpixel iy stays inside column x */
                {
                    xx = x + ix1;
                    yy = y + ycen_offset[x] - ycen_offset[xx];
                    if (xx >= xown0 && xx < xown1)
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy, w);
                }
            }
            dyc[x] = dy;
        }
    }
}

/* Parallel zeta build, chunked over source/target COLUMNS: a source
   column feeds target columns at most the maximum horizontal shift away,
   so each chunk re-processes a halo of source columns around its owned
   target range and every pixel's zeta list is written by exactly one
   chunk in the exact sequential order - the result is bit-identical to
   zeta_tensors() for any chunk or thread count. Each chunk runs its
   columns' recurrences completely, so no state hand-off is needed. */
typedef struct
{
    int ncols, nrows, ny, osample, K, halo, y_lower_lim;
    double *ycen;
    int *ycen_offset;
    double *slitcurve, *slitdeltas;
    zeta_ref *zeta;
    int *m_zeta;
    zeta_rng *z_rng;
    _Atomic int fail;
} zbuild_ctx;

static void zeta_build_chunk(void *p, int k)
{
    zbuild_ctx *c = p;
    const int ncols = c->ncols, nrows = c->nrows, osample = c->osample;
    const double step = 1.e0 / osample;
    const int x0 = (int)((size_t)k * ncols / c->K);
    const int x1 = (int)((size_t)(k + 1) * ncols / c->K);
    int x, yy;
    if (x0 >= x1)
        return;
    /* clear the owned columns of the per-pixel bookkeeping */
    for (yy = 0; yy < nrows; yy++)
        for (x = x0; x < x1; x++)
        {
            const size_t i = x + (size_t)yy * ncols;
            c->m_zeta[i] = 0;
            c->z_rng[i].min_iy = INT_MAX;
            c->z_rng[i].max_iy = INT_MIN;
            c->z_rng[i].min_x = INT_MAX;
            c->z_rng[i].max_x = INT_MIN;
        }
    const int xsrc0 = max(0, x0 - c->halo);
    const int xsrc1 = min(ncols, x1 + c->halo);
    int *iy1c = malloc(ncols * sizeof(int));
    int *iy2c = malloc(ncols * sizeof(int));
    double *dyc = malloc(ncols * sizeof(double));
    double *d1c = malloc(ncols * sizeof(double));
    double *d2c = malloc(ncols * sizeof(double));
    if (!iy1c || !iy2c || !dyc || !d1c || !d2c)
    {
        free(iy1c); free(iy2c); free(dyc); free(d1c); free(d2c);
        atomic_store(&c->fail, 1);
        return;
    }
    for (x = xsrc0; x < xsrc1; x++)
    {
        iy2c[x] = osample - floor(c->ycen[x] * osample);
        iy1c[x] = iy2c[x] - osample;
        d1c[x] = fmod(c->ycen[x], step);
        if (d1c[x] == 0)
            d1c[x] = step;
        d2c[x] = step - d1c[x];
        dyc[x] = c->ycen[x] - floor((c->y_lower_lim + c->ycen[x]) / step) * step - step;
    }
    zeta_build_cols(ncols, nrows, xsrc0, xsrc1, x0, x1, c->ycen,
                    c->ycen_offset, osample, c->slitcurve, c->slitdeltas,
                    iy1c, iy2c, d1c, d2c, dyc, c->zeta, c->m_zeta, c->z_rng);
    free(iy1c);
    free(iy2c);
    free(dyc);
    free(d1c);
    free(d2c);
}

static int zeta_tensors_par(
    int ncols, int nrows, int ny, double *ycen, int *ycen_offset,
    int y_lower_lim, int osample, double *slitcurve, double *slitdeltas,
    zeta_ref *zeta, int *m_zeta, zeta_rng *z_rng, int delta_x, pool_t *pool)
{
    zbuild_ctx c;
    /* Halo: |shift| <= delta_x for curvature or slitdeltas alone; the sum
       of both is bounded by 2*delta_x, +2 for the split-target column and
       rounding. Chunks that would mostly re-process halo fall back. */
    const int halo = 2 * delta_x + 2;
    int K = min(PAR_CHUNKS, max(1, ncols / (2 * halo)));
    if (K <= 1)
        return zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim,
                            osample, slitcurve, slitdeltas, zeta, m_zeta, z_rng);
    c.ncols = ncols;
    c.nrows = nrows;
    c.ny = ny;
    c.osample = osample;
    c.K = K;
    c.halo = halo;
    c.y_lower_lim = y_lower_lim;
    c.ycen = ycen;
    c.ycen_offset = ycen_offset;
    c.slitcurve = slitcurve;
    c.slitdeltas = slitdeltas;
    c.zeta = zeta;
    c.m_zeta = m_zeta;
    c.z_rng = z_rng;
    atomic_store(&c.fail, 0);
    pool_run(pool, zeta_build_chunk, &c, K);
    if (atomic_load(&c.fail))
        return zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim,
                            osample, slitcurve, slitdeltas, zeta, m_zeta, z_rng);
    return 0;
}


/* ---------- Fast path for exactly flat geometry ----------
   When every horizontal shift is exactly zero (all curvature coefficients
   and all slitdeltas are 0.0), pixel (x,y) receives exactly the subpixels
   iy = k0col[x] + y*osample .. + osample of its own column, with weights
   (d1, step, ..., step, step-d1) where d1 = d1col[x]. The zeta tensor is
   then fully determined by ycen, so the SLE fills, the model and the
   uncertainty pass are computed directly, without building or streaming
   zeta (by far the largest array). All sums are algebraically identical
   to the general path; accumulation order differs only at the rounding
   level (same class of reordering as the Round 1/2 optimizations). */

static void flat_setup(int ncols, int osample, const double *ycen,
                       int *k0col, double *d1col)
{
    const double step = 1.e0 / osample;
    for (int x = 0; x < ncols; x++)
    {
        /* same expressions as in zeta_tensors: k0col[x] is the first
           subpixel of row y=0, i.e. iy1 after the first `iy1 += osample` */
        k0col[x] = osample - floor(ycen[x] * osample);
        double d1 = fmod(ycen[x], step);
        if (d1 == 0)
            d1 = step;
        d1col[x] = d1;
    }
}

/* sL system: one pixel contributes the outer product of
   sP[x] * (d1, s, ..., s, s-d1) at band rows k0..k0+osample. Per detector
   row y and per k0-class c the weight products are polynomials in d1 of
   degree <= 2, so all contributions of one row collapse into 5 moments
   per class: M{0,1,2} = sum mask*sP^2*d1^{0,1,2}, R{0,1} = sum mask*im*sP*d1^{0,1}. */
/* Fills rows y0..y1-1; band-row indices are offset by lo so the target can
   be a partial slice (lo = 0 with the full matrices for the sequential
   call). */
static void flat_fill_sL(int ncols, int y0, int y1, int osample,
                         const double *im, const unsigned char *mask,
                         const double *sP, const int *k0col, const double *d1col,
                         double *scratch, int lo, double *l_Aij, double *l_bj)
{
    const double s = 1.e0 / osample;
    const int bw = 4 * osample + 1;
    const int nc = osample + 1; /* classes c = 1..osample, indexed directly */
    /* Four interleaved accumulator sets (combined per row) break the
       serial dependency chains through the per-class sums; this only
       reorders the additions at the rounding level. */
    double *M0 = scratch, *M1 = M0 + 4 * nc, *M2 = M1 + 4 * nc,
           *R0 = M2 + 4 * nc, *R1 = R0 + 4 * nc;

    for (int y = y0; y < y1; y++)
    {
        for (int c = 0; c < 4 * nc; c++)
            M0[c] = M1[c] = M2[c] = R0[c] = R1[c] = 0.e0;
        const double *imrow = im + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const int c = k0col[x] + nc * (x & 3);
            const double d = d1col[x];
            const double a = sP[x] * sP[x];
            const double b = imrow[x] * sP[x];
            M0[c] += a;
            M1[c] += a * d;
            M2[c] += a * d * d;
            R0[c] += b;
            R1[c] += b * d;
        }
        for (int c = 1; c <= osample; c++)
        {
            M0[c] = ((M0[c] + M0[c + nc]) + M0[c + 2 * nc]) + M0[c + 3 * nc];
            M1[c] = ((M1[c] + M1[c + nc]) + M1[c + 2 * nc]) + M1[c + 3 * nc];
            M2[c] = ((M2[c] + M2[c + nc]) + M2[c + 2 * nc]) + M2[c + 3 * nc];
            R0[c] = ((R0[c] + R0[c + nc]) + R0[c + 2 * nc]) + R0[c + 3 * nc];
            R1[c] = ((R1[c] + R1[c + nc]) + R1[c + 2 * nc]) + R1[c + 3 * nc];
        }
        for (int c = 1; c <= osample; c++)
        {
            if (M0[c] == 0.e0 && R0[c] == 0.e0)
                continue;
            const int k0 = y * osample + c - lo;
            const double m0ss = M0[c] * s * s;
            const double m1s = M1[c] * s;
            const double r0s = R0[c] * s;
            /* band row k0 (first subpixel, weight d1): pairs with all */
            double *arow = &l_Aij[(size_t)k0 * bw + 2 * osample];
            arow[0] += M2[c];                    /* d1*d1        */
            for (int d = 1; d < osample; d++)
                arow[d] += m1s;                  /* d1*s         */
            arow[osample] += m1s - M2[c];        /* d1*(s-d1)    */
            l_bj[k0] += R1[c];
            /* band rows k0+i (interior, weight s) */
            for (int i = 1; i < osample; i++)
            {
                arow = &l_Aij[(size_t)(k0 + i) * bw + 2 * osample];
                for (int d = 0; d < osample - i; d++)
                    arow[d] += m0ss;             /* s*s          */
                arow[osample - i] += m0ss - m1s; /* s*(s-d1)     */
                l_bj[k0 + i] += r0s;
            }
            /* band row k0+osample (last subpixel, weight s-d1) */
            l_Aij[(size_t)(k0 + osample) * bw + 2 * osample]
                += m0ss - 2 * m1s + M2[c];       /* (s-d1)^2     */
            l_bj[k0 + osample] += r0s - R1[c];
        }
    }
}

/* Per-row, per-class pieces of the merged slit-function value
   v(x,y) = sum_i w_i(x) * sL[k0+i] = A(y,c) + d1col[x] * B(y,c) */
static void flat_row_AB(int y, int osample, const double *sL,
                        double *A, double *B)
{
    const double s = 1.e0 / osample;
    for (int c = 1; c <= osample; c++)
    {
        const int k0 = y * osample + c;
        double mid = 0.e0;
        for (int i = 1; i < osample; i++)
            mid += sL[k0 + i];
        A[c] = s * (mid + sL[k0 + osample]);
        B[c] = sL[k0] - sL[k0 + osample];
    }
}

/* sP system: with all shifts zero every pixel maps to its own column, so
   the matrix is purely diagonal (band offset 2*delta_x; delta_x can still
   be 1 when lambda_sP > 0 forces a minimum band). */
static void flat_fill_sP(int ncols, int nrows, int osample, int delta_x, int nx,
                         const double *im, const unsigned char *mask,
                         const double *sL, const int *k0col, const double *d1col,
                         double *scratch, double *p_Aij, double *p_bj)
{
    double *A = scratch, *B = A + osample + 1;
    for (int y = 0; y < nrows; y++)
    {
        flat_row_AB(y, osample, sL, A, B);
        const double *imrow = im + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const double v = A[k0col[x]] + d1col[x] * B[k0col[x]];
            p_Aij[(size_t)x * nx + 2 * delta_x] += v * v;
            p_bj[x] += imrow[x] * v;
        }
    }
}

static void flat_model(int ncols, int y0, int y1, int osample,
                       const double *sP, const double *sL,
                       const int *k0col, const double *d1col,
                       double *scratch, double *model)
{
    double *A = scratch, *B = A + osample + 1;
    for (int y = y0; y < y1; y++)
    {
        flat_row_AB(y, osample, sL, A, B);
        double *mdrow = model + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
            mdrow[x] = sP[x] * (A[k0col[x]] + d1col[x] * B[k0col[x]]);
    }
}

/* Uncertainty pass: per pixel the summed weight and summed squared weight
   of its (fixed) subpixel list, precomputed per column into swcol/sw2col. */
static void flat_unc(int ncols, int nrows, int nx,
                     const double *im, const double *model,
                     const unsigned char *mask,
                     const double *swcol, const double *sw2col,
                     double *unc, double *p_bj, double *p_Aij)
{
    for (int y = 0; y < nrows; y++)
    {
        const double *imrow = im + (size_t)y * ncols;
        const double *mdrow = model + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const double tmp = imrow[x] - mdrow[x];
            unc[x] += tmp * tmp * swcol[x];
            p_bj[x] += swcol[x];
            p_Aij[(size_t)x * nx] += sw2col[x];
        }
    }
}


/* ---------- Parallel chunk workers for the SLE fills and the model ----------
   Each chunk accumulates into its own partial band slice (rows [lo..hi] of
   the full matrix), allocated here and reduced + freed by the caller in
   chunk order. The loop bodies are the same as the sequential fills in
   slitdec(); only the accumulation target is redirected. */
typedef struct
{
    int ncols, nrows, ny, osample, delta_x, nx;
    const double *im;
    const unsigned char *mask;
    const double *sP, *sL;
    const zeta_ref *zeta;
    const int *m_zeta;
    const zeta_rng *z_rng;
    double *model;
    double *slice[PAR_CHUNKS];
    double *bslice[PAR_CHUNKS];
    int lo[PAR_CHUNKS];
    int hi[PAR_CHUNKS];
    _Atomic int fail;
    double *zwall;
    int *zkall;
    int zbuf;
    /* flat fast path */
    const int *k0col;
    const double *d1col;
    double *fscratch; /* PAR_CHUNKS x 5*(osample+1) */
    /* uncertainty pass */
    double *uslice[PAR_CHUNKS];
    /* dev / mask-adjust pass */
    double devsum[PAR_CHUNKS];
    long devcnt[PAR_CHUNKS];
    double thresh;
    unsigned char *mask_out;
} sle_ctx;

#define CH_ZIDX(c, x, y, z) ((z) + (x) * (3 * ((c)->osample + 1)) + (size_t)(y) * (3 * ((c)->osample + 1)) * (c)->ncols)
#define CH_MIDX(c, x, y) ((x) + (size_t)(y) * (c)->ncols)

static void lfill_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, nrows = c->nrows, osample = c->osample;
    const int bw = 4 * osample + 1;
    const int y0 = (int)((size_t)k * nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * nrows / PAR_CHUNKS);
    int xx, yy, m, n, iy;
    int lo = INT_MAX, hi = INT_MIN;

    c->slice[k] = c->bslice[k] = NULL;
    for (yy = y0; yy < y1; yy++)
        for (xx = 0; xx < ncols; xx++)
        {
            if (c->m_zeta[CH_MIDX(c, xx, yy)] <= 0 || !c->mask[CH_MIDX(c, xx, yy)])
                continue;
            const zeta_rng *zr = &c->z_rng[CH_MIDX(c, xx, yy)];
            if (zr->min_iy < lo)
                lo = zr->min_iy;
            if (zr->max_iy > hi)
                hi = zr->max_iy;
        }
    c->lo[k] = lo;
    c->hi[k] = hi;
    if (lo > hi)
        return;
    const int rows = hi - lo + 1;
    double *restrict S = calloc((size_t)rows * bw, sizeof(double));
    double *restrict B = calloc(rows, sizeof(double));
    if (!S || !B)
    {
        free(S);
        free(B);
        atomic_store(&c->fail, 1);
        return;
    }
    c->slice[k] = S;
    c->bslice[k] = B;
    double *zw = c->zwall + (size_t)k * c->zbuf;
    int *zk = c->zkall + (size_t)k * c->zbuf;

    for (yy = y0; yy < y1; yy++)
    {
        for (xx = 0; xx < ncols; xx++)
        {
            const int mz = c->m_zeta[CH_MIDX(c, xx, yy)];
            if (mz <= 0 || !c->mask[CH_MIDX(c, xx, yy)])
                continue;
            const zeta_ref *zrow = &c->zeta[CH_ZIDX(c, xx, yy, 0)];
            const double imv = c->im[CH_MIDX(c, xx, yy)];
            const zeta_rng *zr = &c->z_rng[CH_MIDX(c, xx, yy)];
            const int k0 = zr->min_iy;
            const int rng = zr->max_iy - k0;
            if (rng <= 2 * osample)
            {
                for (n = 0; n <= rng; n++)
                    zw[n] = 0.e0;
                for (m = 0; m < mz; m++)
                    zw[zrow[m].iy - k0] += c->sP[zrow[m].x] * zrow[m].w;
                for (m = 0; m <= rng; m++)
                {
                    const double um = zw[m];
                    const double *restrict uv = zw + m;
                    double *restrict arow = S + (size_t)(k0 - lo + m) * bw + 2 * osample;
                    const int dmax = rng - m;
                    for (n = 0; n <= dmax; n++)
                        arow[n] += um * uv[n];
                    B[k0 - lo + m] += imv * um;
                }
                continue;
            }
            /* over-wide fallback, see the sequential fill */
            int nk = 0;
            for (m = 0; m < mz; m++)
            {
                const int key = zrow[m].iy;
                const double v = c->sP[zrow[m].x] * zrow[m].w;
                for (n = 0; n < nk; n++)
                    if (zk[n] == key)
                    {
                        zw[n] += v;
                        break;
                    }
                if (n == nk)
                {
                    zk[nk] = key;
                    zw[nk++] = v;
                }
            }
            for (m = 0; m < nk; m++)
            {
                iy = zk[m];
                const double um = zw[m];
                S[(size_t)(iy - lo) * bw + 2 * osample] += um * um;
                for (n = m + 1; n < nk; n++)
                {
                    const int iyn = zk[n];
                    const int lo2 = min(iy, iyn);
                    const int d = abs(iyn - iy);
                    if (d > 2 * osample)
                        continue;
                    S[(size_t)(lo2 - lo) * bw + d + 2 * osample] += zw[n] * um;
                }
                B[iy - lo] += imv * um;
            }
        }
    }
}

static void pfill_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, nrows = c->nrows, osample = c->osample;
    const int delta_x = c->delta_x, nx = c->nx;
    const int x0 = (int)((size_t)k * ncols / PAR_CHUNKS);
    const int x1 = (int)((size_t)(k + 1) * ncols / PAR_CHUNKS);
    int xx, yy, m, n, x;
    int lo = INT_MAX, hi = INT_MIN;

    c->slice[k] = c->bslice[k] = NULL;
    for (yy = 0; yy < nrows; yy++)
        for (xx = x0; xx < x1; xx++)
        {
            if (c->m_zeta[CH_MIDX(c, xx, yy)] <= 0 || !c->mask[CH_MIDX(c, xx, yy)])
                continue;
            const zeta_rng *zr = &c->z_rng[CH_MIDX(c, xx, yy)];
            if (zr->min_x < lo)
                lo = zr->min_x;
            if (zr->max_x > hi)
                hi = zr->max_x;
        }
    c->lo[k] = lo;
    c->hi[k] = hi;
    if (lo > hi)
        return;
    const int rows = hi - lo + 1;
    double *restrict S = calloc((size_t)rows * nx, sizeof(double));
    double *restrict B = calloc(rows, sizeof(double));
    if (!S || !B)
    {
        free(S);
        free(B);
        atomic_store(&c->fail, 1);
        return;
    }
    c->slice[k] = S;
    c->bslice[k] = B;
    double *zw = c->zwall + (size_t)k * c->zbuf;
    int *zk = c->zkall + (size_t)k * c->zbuf;

    for (yy = 0; yy < nrows; yy++)
    {
        for (xx = x0; xx < x1; xx++)
        {
            const int mz = c->m_zeta[CH_MIDX(c, xx, yy)];
            if (mz <= 0 || !c->mask[CH_MIDX(c, xx, yy)])
                continue;
            const zeta_ref *zrow = &c->zeta[CH_ZIDX(c, xx, yy, 0)];
            const double imv = c->im[CH_MIDX(c, xx, yy)];
            const zeta_rng *zr = &c->z_rng[CH_MIDX(c, xx, yy)];
            const int k0 = zr->min_x;
            const int rng = zr->max_x - k0;
            if (rng <= 2 * delta_x)
            {
                for (n = 0; n <= rng; n++)
                    zw[n] = 0.e0;
                for (m = 0; m < mz; m++)
                    zw[zrow[m].x - k0] += c->sL[zrow[m].iy] * zrow[m].w;
                for (m = 0; m <= rng; m++)
                {
                    const double um = zw[m];
                    const double *restrict uv = zw + m;
                    double *restrict arow = S + (size_t)(k0 - lo + m) * nx + 2 * delta_x;
                    const int dmax = rng - m;
                    for (n = 0; n <= dmax; n++)
                        arow[n] += um * uv[n];
                    B[k0 - lo + m] += imv * um;
                }
                continue;
            }
            /* over-wide fallback */
            int nk = 0;
            for (m = 0; m < mz; m++)
            {
                const int key = zrow[m].x;
                const double v = c->sL[zrow[m].iy] * zrow[m].w;
                for (n = 0; n < nk; n++)
                    if (zk[n] == key)
                    {
                        zw[n] += v;
                        break;
                    }
                if (n == nk)
                {
                    zk[nk] = key;
                    zw[nk++] = v;
                }
            }
            for (m = 0; m < nk; m++)
            {
                x = zk[m];
                const double um = zw[m];
                S[(size_t)(x - lo) * nx + 2 * delta_x] += um * um;
                for (n = m + 1; n < nk; n++)
                {
                    const int xn = zk[n];
                    const int lo2 = min(x, xn);
                    const int d = abs(xn - x);
                    if (d > 2 * delta_x)
                        continue;
                    S[(size_t)(lo2 - lo) * nx + d + 2 * delta_x] += zw[n] * um;
                }
                B[x - lo] += imv * um;
            }
        }
    }
}

static void model_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, nrows = c->nrows;
    const int y0 = (int)((size_t)k * nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * nrows / PAR_CHUNKS);
    int xx, yy, m;
    for (yy = y0; yy < y1; yy++)
    {
        for (xx = 0; xx < ncols; xx++)
        {
            const zeta_ref *zrow = &c->zeta[CH_ZIDX(c, xx, yy, 0)];
            const int mz = c->m_zeta[CH_MIDX(c, xx, yy)];
            double acc = 0.;
            for (m = 0; m < mz; m++)
                acc += c->sP[zrow[m].x] * c->sL[zrow[m].iy] * zrow[m].w;
            c->model[CH_MIDX(c, xx, yy)] = acc;
        }
    }
}

/* Reduce per-chunk partial slices into the full band matrix + RHS,
   in chunk order (deterministic), freeing the slices. */
static void sle_reduce(sle_ctx *c, int bw, double *Aij, double *bj)
{
    for (int k = 0; k < PAR_CHUNKS; k++)
    {
        if (!c->slice[k])
            continue;
        const int lo = c->lo[k];
        const int rows = c->hi[k] - lo + 1;
        double *restrict dst = Aij + (size_t)lo * bw;
        const double *restrict s = c->slice[k];
        for (size_t i = 0; i < (size_t)rows * bw; i++)
            dst[i] += s[i];
        double *restrict db = bj + lo;
        const double *restrict sb = c->bslice[k];
        for (int i = 0; i < rows; i++)
            db[i] += sb[i];
        free(c->slice[k]);
        free(c->bslice[k]);
        c->slice[k] = c->bslice[k] = NULL;
    }
}

static void sle_free_slices(sle_ctx *c)
{
    for (int k = 0; k < PAR_CHUNKS; k++)
    {
        free(c->slice[k]);
        free(c->bslice[k]);
        c->slice[k] = c->bslice[k] = NULL;
    }
}

static void flat_lfill_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int osample = c->osample;
    const int bw = 4 * osample + 1;
    const int y0 = (int)((size_t)k * c->nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * c->nrows / PAR_CHUNKS);
    c->slice[k] = c->bslice[k] = NULL;
    if (y0 >= y1)
        return;
    /* rows y0..y1-1 write band rows y*osample + c + i, c in 1..osample,
       i in 0..osample */
    const int lo = y0 * osample + 1;
    const int hi = (y1 + 1) * osample;
    const int rows = hi - lo + 1;
    double *S = calloc((size_t)rows * bw, sizeof(double));
    double *B = calloc(rows, sizeof(double));
    if (!S || !B)
    {
        free(S);
        free(B);
        atomic_store(&c->fail, 1);
        return;
    }
    c->slice[k] = S;
    c->bslice[k] = B;
    c->lo[k] = lo;
    c->hi[k] = hi;
    flat_fill_sL(c->ncols, y0, y1, osample, c->im, c->mask, c->sP,
                 c->k0col, c->d1col, c->fscratch + (size_t)k * 20 * (osample + 1),
                 lo, S, B);
}

static void flat_pfill_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, osample = c->osample;
    const int y0 = (int)((size_t)k * c->nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * c->nrows / PAR_CHUNKS);
    c->slice[k] = c->bslice[k] = NULL;
    if (y0 >= y1)
        return;
    double *D = calloc(ncols, sizeof(double));
    double *B = calloc(ncols, sizeof(double));
    if (!D || !B)
    {
        free(D);
        free(B);
        atomic_store(&c->fail, 1);
        return;
    }
    c->slice[k] = D;
    c->bslice[k] = B;
    double *ab = c->fscratch + (size_t)k * 20 * (osample + 1);
    double *A = ab, *Bc = ab + osample + 1;
    for (int y = y0; y < y1; y++)
    {
        flat_row_AB(y, osample, c->sL, A, Bc);
        const double *imrow = c->im + (size_t)y * ncols;
        const unsigned char *mrow = c->mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const double v = A[c->k0col[x]] + c->d1col[x] * Bc[c->k0col[x]];
            D[x] += v * v;
            B[x] += imrow[x] * v;
        }
    }
}

static void flat_model_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int y0 = (int)((size_t)k * c->nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * c->nrows / PAR_CHUNKS);
    if (y0 >= y1)
        return;
    flat_model(c->ncols, y0, y1, c->osample, c->sP, c->sL, c->k0col, c->d1col,
               c->fscratch + (size_t)k * 20 * (c->osample + 1), c->model);
}

/* Residual RMS accumulation (dev) and the mask threshold pass, both over
   x in [delta_x, ncols - delta_x). Partial sums per chunk, reduced in
   chunk order. */
static void dev_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, delta_x = c->delta_x;
    const int y0 = (int)((size_t)k * c->nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * c->nrows / PAR_CHUNKS);
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    long cnt = 0;
    for (int y = y0; y < y1; y++)
    {
        const double *imrow = c->im + (size_t)y * ncols;
        const double *mdrow = c->model + (size_t)y * ncols;
        const unsigned char *mrow = c->mask + (size_t)y * ncols;
        /* four partial sums break the dependency chain (rounding-level) */
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (int x = delta_x; x < ncols - delta_x; x++)
        {
            if (mrow[x])
            {
                const double resid = mdrow[x] - imrow[x];
                const double r2 = resid * resid;
                switch (x & 3)
                {
                case 0: t0 += r2; break;
                case 1: t1 += r2; break;
                case 2: t2 += r2; break;
                default: t3 += r2; break;
                }
                cnt++;
            }
        }
        s0 += t0;
        s1 += t1;
        s2 += t2;
        s3 += t3;
    }
    c->devsum[k] = ((s0 + s1) + s2) + s3;
    c->devcnt[k] = cnt;
}

/* Uncertainty accumulation (general path): per-chunk partial sums of
   residual^2 * w, w, and w^2 per source column. */
static void unc_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, nrows = c->nrows;
    const int y0 = (int)((size_t)k * nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * nrows / PAR_CHUNKS);
    int x, y, m;
    c->uslice[k] = NULL;
    if (y0 >= y1)
        return;
    double *U = calloc((size_t)3 * ncols, sizeof(double));
    if (!U)
    {
        atomic_store(&c->fail, 1);
        return;
    }
    c->uslice[k] = U;
    double *restrict us = U, *restrict pb = U + ncols, *restrict pa = U + 2 * ncols;
    for (y = y0; y < y1; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            const size_t i = x + (size_t)y * ncols;
            if (!c->mask[i])
                continue;
            const zeta_ref *zrow = &c->zeta[CH_ZIDX(c, x, y, 0)];
            const int mz = c->m_zeta[i];
            const double tmp = c->im[i] - c->model[i];
            for (m = 0; m < mz; m++)
            {
                const int xx = zrow[m].x;
                const double ww = zrow[m].w;
                us[xx] += tmp * tmp * ww;
                pb[xx] += ww;
                pa[xx] += ww * ww;
            }
        }
    }
}

static void maskadj_chunk(void *p, int k)
{
    sle_ctx *c = p;
    const int ncols = c->ncols, delta_x = c->delta_x;
    const int y0 = (int)((size_t)k * c->nrows / PAR_CHUNKS);
    const int y1 = (int)((size_t)(k + 1) * c->nrows / PAR_CHUNKS);
    const double thresh = c->thresh;
    for (int y = y0; y < y1; y++)
    {
        const double *imrow = c->im + (size_t)y * ncols;
        const double *mdrow = c->model + (size_t)y * ncols;
        unsigned char *mrow = c->mask_out + (size_t)y * ncols;
        for (int x = delta_x; x < ncols - delta_x; x++)
            mrow[x] = fabs(mdrow[x] - imrow[x]) < thresh;
    }
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
                    double lambda_sP,
                    double lambda_sL,
                    int maxiter,
                    double kappa,
                    int use_preset,
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
        Individual pixel uncertainties. Currently unused: the output
        uncertainties are estimated from the data - model residuals.
    mask : byte array of shape (nrows, ncols)
        Initial and final mask for the swath, both in and output
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary.
        Should only contain values between 0 and 1.
    slitcurve : double array of shape (ncols, 6)
        Slit curvature polynomial coefficients c0..c5 for each column
    slitdeltas : double array of shape (ny,)
        Additional per-subpixel horizontal offsets
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
    int x, xx, y, yy, iy, n, m, nx, ny;
    double norm, dev, lambda, diag_tot, ww, tmp;
    double sP_change, sP_stop, sP_med;
    int iter, delta_x;
    unsigned int isum;
    int *ycen_offset;
    int y_lower_lim = nrows / 2;

    // For the solving of the equation system
    double *l_Aij, *l_bj, *p_Aij, *p_bj;
    double *sP_old, *sP_diff;
    // Scratch buffers for per-pixel merged zeta weights (mz <= 3 * (osample + 1))
    double *zw;
    int *zk;

    // For the geometry
    zeta_ref *zeta = NULL;
    int *m_zeta = NULL;
    zeta_rng *z_rng = NULL;

    // Flat-geometry fast path (see the flat_* helpers above)
    int fast_flat;
    int *k0col = NULL;
    double *d1col = NULL, *flat_scratch = NULL, *swcol = NULL, *sw2col = NULL;

    // Parallel fills (see the chunk workers above)
    const int nthreads = par_nthreads(ncols, nrows);
    pool_t *pool = NULL;
    sle_ctx pc;
    memset(&pc, 0, sizeof pc);

    // The Optimization results
    double success, status;

    // maxiter = 20; // Maximum number of iterations
    sP_stop = 5e-5;  // Convergence threshold: 99th percentile spectrum change relative to median
    success = 1;
    status = 0;

    sP_change = INFINITY;
    ny = osample * (nrows + 1) + 1; /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */

#if DEBUG
    _ncols = ncols;
    _nrows = nrows;
    _ny = ny;
    _osample = osample;
    printf("ncols: %d, nrows: %d, ny: %d, osample: %d\n", _ncols, _nrows, _ny, _osample);
#endif

    // If we want to smooth the spectrum we need at least delta_x = 1
    // Otherwise delta_x = 0 works if there is no curvature
    delta_x = lambda_sP == 0 ? 0 : 1;
    for (x = 0; x < ncols; x++)
    {
        /* all-zero polynomial contributes ceil(0) = 0: skip the row scan */
        if (slitcurve[curve_index(x, 1)] == 0.e0 &&
            slitcurve[curve_index(x, 2)] == 0.e0 &&
            slitcurve[curve_index(x, 3)] == 0.e0 &&
            slitcurve[curve_index(x, 4)] == 0.e0 &&
            slitcurve[curve_index(x, 5)] == 0.e0)
            continue;
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

    nx = 4 * delta_x + 1; /* Maximum horizontal shift in detector pixels due to slit image curvature         */

#if DEBUG
    _nx = nx;
#endif

    // The curvature is larger than the number of columns
    // Usually that means that the curvature is messed up
    if (nx > ncols)
    {
        info[0] = 0;        //failed
        info[1] = sP_change; //INFINITY
        info[2] = -2;       // curvature to large
        info[3] = 0;
        info[4] = delta_x;
        return -1;
    }

    /* The fast path applies when the geometry is exactly flat: all
       curvature coefficients and all slitdeltas are 0.0, so every
       subpixel shift is exactly zero. (delta_x can still be 1 when
       lambda_sP > 0 forces a minimum band width.) */
    fast_flat = 1;
    for (x = 0; x < ncols && fast_flat; x++)
        for (m = 1; m < 6; m++)
            if (slitcurve[curve_index(x, m)] != 0.e0)
            {
                fast_flat = 0;
                break;
            }
    if (fast_flat)
        for (iy = 0; iy < ny; iy++)
            if (slitdeltas[iy] != 0.e0)
            {
                fast_flat = 0;
                break;
            }

    l_Aij = malloc(MAX_LAIJ * sizeof(double));
    p_Aij = malloc(MAX_PAIJ * sizeof(double));
    l_bj = malloc(MAX_LBJ * sizeof(double));
    p_bj = malloc(MAX_PBJ * sizeof(double));
    /* Scratch buffers for per-pixel merged zeta weights: large enough for
       both the slit-function window (2*osample+1 <= MAX_ZETA_Z) and the
       spectrum window (2*delta_x+1 <= nx) */
    int zbuf = max(MAX_ZETA_Z, nx);
    zw = malloc(zbuf * sizeof(double));
    zk = malloc(zbuf * sizeof(int));
    if (fast_flat)
    {
        k0col = malloc(ncols * sizeof(int));
        d1col = malloc(ncols * sizeof(double));
        swcol = malloc(ncols * sizeof(double));
        sw2col = malloc(ncols * sizeof(double));
        flat_scratch = malloc(20 * (osample + 1) * sizeof(double));
        if (nthreads > 1)
            pc.fscratch = malloc((size_t)PAR_CHUNKS * 20 * (osample + 1) * sizeof(double));
    }
    else
    {
        zeta = malloc(MAX_ZETA * sizeof(zeta_ref));
        m_zeta = malloc(MAX_MZETA * sizeof(int));
        z_rng = malloc(MAX_MZETA * sizeof(zeta_rng));
        if (nthreads > 1)
        {
            pc.zwall = malloc((size_t)PAR_CHUNKS * zbuf * sizeof(double));
            pc.zkall = malloc((size_t)PAR_CHUNKS * zbuf * sizeof(int));
        }
    }
    pc.ncols = ncols;
    pc.nrows = nrows;
    pc.ny = ny;
    pc.osample = osample;
    pc.delta_x = delta_x;
    pc.nx = nx;
    pc.im = im;
    pc.zbuf = zbuf;
    if (nthreads > 1)
        pool = pool_start(nthreads);
    ycen_offset = malloc(ncols * sizeof(int));
    sP_old = malloc(ncols * sizeof(double));
    sP_diff = malloc(ncols * sizeof(double));

        // remove integer values from ycen, put into ycen_offset
    for (x = 0; x < ncols; x++)
    {
        ycen_offset[x] = ycen[x];
        ycen[x] = ycen[x] - ycen_offset[x];
    }
    
    if (fast_flat)
        flat_setup(ncols, osample, ycen, k0col, d1col);
    else if (nthreads > 1)
        zeta_tensors_par(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas, zeta, m_zeta, z_rng, delta_x, pool);
    else
        zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas, zeta, m_zeta, z_rng);

    /* Preset slit function: the caller supplies sL and we skip solving for
       it. Normalize the preset once here (to sum osample) so callers do not
       have to; inside the loop sL is then left untouched. */
    if (use_preset)
    {
        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
            norm += sL[sl_index(iy)];
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[sl_index(iy)] /= norm;
    }

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    do
    {
        /* Compute slit function sL (skipped when a preset sL is supplied) */
        if (!use_preset)
        {

        /* Prepare the RHS and the matrix */
        for (iy = 0; iy < MAX_LBJ; iy++)
            l_bj[lbj_index(iy)] = 0.e0; /* Clean RHS */
        for (iy = 0; iy < MAX_LAIJ; iy++)
            l_Aij[iy] = 0;

        /* Fill in SLE arrays for slit function.
           Both SLE matrices are sums over detector pixels of all pairs of
           subpixels contributing to that pixel, i.e. pairs of entries in the
           pixel's zeta list. Iterating pixel-centrically reads zeta
           sequentially and skips masked pixels entirely. Accumulation order
           differs from the historic xi-centric loop only at the rounding
           level. */
        if (fast_flat && nthreads > 1)
        {
            pc.mask = mask;
            pc.sP = sP;
            pc.k0col = k0col;
            pc.d1col = d1col;
            atomic_store(&pc.fail, 0);
            pool_run(pool, flat_lfill_chunk, &pc, PAR_CHUNKS);
            if (atomic_load(&pc.fail))
            {
                sle_free_slices(&pc);
                for (iy = 0; iy < MAX_LBJ; iy++)
                    l_bj[lbj_index(iy)] = 0.e0;
                for (iy = 0; iy < MAX_LAIJ; iy++)
                    l_Aij[iy] = 0;
                flat_fill_sL(ncols, 0, nrows, osample, im, mask, sP, k0col, d1col,
                             flat_scratch, 0, l_Aij, l_bj);
            }
            else
                sle_reduce(&pc, 4 * osample + 1, l_Aij, l_bj);
        }
        else if (fast_flat)
            flat_fill_sL(ncols, 0, nrows, osample, im, mask, sP, k0col, d1col,
                         flat_scratch, 0, l_Aij, l_bj);
        else if (nthreads > 1)
        {
            pc.mask = mask;
            pc.sP = sP;
            pc.sL = sL;
            pc.zeta = zeta;
            pc.m_zeta = m_zeta;
            pc.z_rng = z_rng;
            atomic_store(&pc.fail, 0);
            pool_run(pool, lfill_chunk, &pc, PAR_CHUNKS);
            if (atomic_load(&pc.fail))
            {
                /* allocation failure in a chunk: redo sequentially */
                sle_free_slices(&pc);
                for (iy = 0; iy < MAX_LBJ; iy++)
                    l_bj[lbj_index(iy)] = 0.e0;
                for (iy = 0; iy < MAX_LAIJ; iy++)
                    l_Aij[iy] = 0;
                goto lfill_sequential;
            }
            sle_reduce(&pc, 4 * osample + 1, l_Aij, l_bj);
        }
        else
        lfill_sequential:
        /* Row-outer fill: the band rows touched by one detector row span
           only ~2*osample rows of l_Aij, which then stay L1-resident for
           the whole row sweep; iterated column-outer the walk swept the
           entire band matrix once per column (measured ~6x slower on the
           pair loop). zeta is y-major, so this order also reads it
           sequentially. */
        for (yy = 0; yy < nrows; yy++)
        {
            for (xx = 0; xx < ncols; xx++)
            {
                const int mz = m_zeta[mzeta_index(xx, yy)];
                if (mz <= 0 || !mask[im_index(xx, yy)])
                    continue;
                const zeta_ref *zrow = &zeta[zeta_index(xx, yy, 0)];
                const double imv = im[im_index(xx, yy)];
                /* Merge entries sharing the same subpixel index iy: only the
                   summed weight enters both the matrix and the RHS. The iy of
                   one pixel span at most 2*osample+1 indices (the band width
                   assumed by the matrix), so merge into a dense window
                   zw[iy - k0] instead of searching a list of unique keys. */
                const zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
                const int k0 = zr->min_iy;
                const int rng = zr->max_iy - k0;
                if (rng <= 2 * osample)
                {
                    for (n = 0; n <= rng; n++)
                        zw[n] = 0.e0;
                    for (m = 0; m < mz; m++)
                        zw[zrow[m].iy - k0] += sP[sp_index(zrow[m].x)] * zrow[m].w;
                    /* The matrix is symmetric: accumulate each unordered pair
                       once into the upper bands (mirrored below after the
                       fill). Walking the window row-wise makes the inner loop
                       contiguous in both operands; window entries between
                       actual keys are zero and add exactly nothing. */
                    for (m = 0; m <= rng; m++)
                    {
                        const double um = zw[m];
                        const double *restrict uv = zw + m;
                        double *restrict arow = &l_Aij[laij_index(k0 + m, 2 * osample)];
                        const int dmax = rng - m;
                        for (n = 0; n <= dmax; n++)
                            arow[n] += um * uv[n];
                        l_bj[lbj_index(k0 + m)] += imv * um;
                    }
                    continue;
                }
                /* Over-wide list (extreme geometry): merge by searching
                   unique keys, as before */
                int nk = 0;
                for (m = 0; m < mz; m++)
                {
                    const int key = zrow[m].iy;
                    const double v = sP[sp_index(zrow[m].x)] * zrow[m].w;
                    for (n = 0; n < nk; n++)
                    {
                        if (zk[n] == key)
                        {
                            zw[n] += v;
                            break;
                        }
                    }
                    if (n == nk)
                    {
                        zk[nk] = key;
                        zw[nk++] = v;
                    }
                }
                for (m = 0; m < nk; m++)
                {
                    iy = zk[m];
                    const double um = zw[m];
                    l_Aij[laij_index(iy, 2 * osample)] += um * um;
                    for (n = m + 1; n < nk; n++)
                    {
                        const int iyn = zk[n];
                        const int lo = min(iy, iyn);
                        const int d = abs(iyn - iy);
                        /* Pairs beyond the band cannot be represented by the
                           band matrix; storing them would write into the next
                           row's band (or past the array). Drop them. */
                        if (d > 2 * osample)
                            continue;
                        l_Aij[laij_index(lo, d + 2 * osample)] += zw[n] * um;
                    }
                    l_bj[lbj_index(iy)] += imv * um;
                }
            }
        }

        /* Mirror the upper bands into the lower bands: A[r+d, 2o-d] = A[r, 2o+d] */
        for (m = 1; m <= 2 * osample; m++)
            for (iy = 0; iy < ny - m; iy++)
                l_Aij[laij_index(iy + m, 2 * osample - m)] = l_Aij[laij_index(iy, 2 * osample + m)];

        diag_tot = 0.e0;
        for (iy = 0; iy < ny; iy++)
            diag_tot += l_Aij[laij_index(iy, 2 * osample)];

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

        } /* end if (!use_preset) */

        /* Compute spectrum sP */
        for (x = 0; x < MAX_PBJ; x++)
            p_bj[pbj_index(x)] = 0;
        for (x = 0; x < MAX_PAIJ; x++)
            p_Aij[x] = 0;

        /* Pixel-centric fill, see comment at the slit function SLE above */
        if (fast_flat && nthreads > 1)
        {
            pc.mask = mask;
            pc.sL = sL;
            pc.k0col = k0col;
            pc.d1col = d1col;
            atomic_store(&pc.fail, 0);
            pool_run(pool, flat_pfill_chunk, &pc, PAR_CHUNKS);
            if (atomic_load(&pc.fail))
            {
                sle_free_slices(&pc);
                for (x = 0; x < MAX_PBJ; x++)
                    p_bj[pbj_index(x)] = 0;
                for (x = 0; x < MAX_PAIJ; x++)
                    p_Aij[x] = 0;
                flat_fill_sP(ncols, nrows, osample, delta_x, nx, im, mask, sL,
                             k0col, d1col, flat_scratch, p_Aij, p_bj);
            }
            else
            {
                for (m = 0; m < PAR_CHUNKS; m++)
                {
                    if (!pc.slice[m])
                        continue;
                    for (x = 0; x < ncols; x++)
                    {
                        p_Aij[paij_index(x, 2 * delta_x)] += pc.slice[m][x];
                        p_bj[pbj_index(x)] += pc.bslice[m][x];
                    }
                    free(pc.slice[m]);
                    free(pc.bslice[m]);
                    pc.slice[m] = pc.bslice[m] = NULL;
                }
            }
        }
        else if (fast_flat)
            flat_fill_sP(ncols, nrows, osample, delta_x, nx, im, mask, sL,
                         k0col, d1col, flat_scratch, p_Aij, p_bj);
        else if (nthreads > 1)
        {
            pc.mask = mask;
            pc.sP = sP;
            pc.sL = sL;
            pc.zeta = zeta;
            pc.m_zeta = m_zeta;
            pc.z_rng = z_rng;
            atomic_store(&pc.fail, 0);
            pool_run(pool, pfill_chunk, &pc, PAR_CHUNKS);
            if (atomic_load(&pc.fail))
            {
                sle_free_slices(&pc);
                for (x = 0; x < MAX_PBJ; x++)
                    p_bj[pbj_index(x)] = 0;
                for (x = 0; x < MAX_PAIJ; x++)
                    p_Aij[x] = 0;
                goto pfill_sequential;
            }
            sle_reduce(&pc, nx, p_Aij, p_bj);
        }
        else
        pfill_sequential:
        /* Row-outer fill, see the sL fill above. For this system the band
           rows are keyed by source column (near xx), so the row sweep also
           walks p_Aij near-sequentially. */
        for (yy = 0; yy < nrows; yy++)
        {
            for (xx = 0; xx < ncols; xx++)
            {
                const int mz = m_zeta[mzeta_index(xx, yy)];
                if (mz <= 0 || !mask[im_index(xx, yy)])
                    continue;
                const zeta_ref *zrow = &zeta[zeta_index(xx, yy, 0)];
                const double imv = im[im_index(xx, yy)];
                /* Merge entries sharing the same source column x; with small
                   curvature this collapses the list to just a few entries.
                   Sources span at most 2*delta_x+1 columns (the band width of
                   the matrix), so merge into a dense window zw[x - k0]. */
                const zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
                const int k0 = zr->min_x;
                const int rng = zr->max_x - k0;
                if (rng <= 2 * delta_x)
                {
                    for (n = 0; n <= rng; n++)
                        zw[n] = 0.e0;
                    for (m = 0; m < mz; m++)
                        zw[zrow[m].x - k0] += sL[sl_index(zrow[m].iy)] * zrow[m].w;
                    /* Symmetric matrix: upper bands only, mirrored after the
                       fill. Window entries between keys are zero. */
                    for (m = 0; m <= rng; m++)
                    {
                        const double um = zw[m];
                        const double *restrict uv = zw + m;
                        double *restrict arow = &p_Aij[paij_index(k0 + m, 2 * delta_x)];
                        const int dmax = rng - m;
                        for (n = 0; n <= dmax; n++)
                            arow[n] += um * uv[n];
                        p_bj[pbj_index(k0 + m)] += imv * um;
                    }
                    continue;
                }
                /* Over-wide list: merge by searching unique keys */
                int nk = 0;
                for (m = 0; m < mz; m++)
                {
                    const int key = zrow[m].x;
                    const double v = sL[sl_index(zrow[m].iy)] * zrow[m].w;
                    for (n = 0; n < nk; n++)
                    {
                        if (zk[n] == key)
                        {
                            zw[n] += v;
                            break;
                        }
                    }
                    if (n == nk)
                    {
                        zk[nk] = key;
                        zw[nk++] = v;
                    }
                }
                for (m = 0; m < nk; m++)
                {
                    x = zk[m];
                    const double um = zw[m];
                    p_Aij[paij_index(x, 2 * delta_x)] += um * um;
                    for (n = m + 1; n < nk; n++)
                    {
                        const int xn = zk[n];
                        const int lo = min(x, xn);
                        const int d = abs(xn - x);
                        /* See the sL fallback: out-of-band pairs are dropped */
                        if (d > 2 * delta_x)
                            continue;
                        p_Aij[paij_index(lo, d + 2 * delta_x)] += zw[n] * um;
                    }
                    p_bj[pbj_index(x)] += imv * um;
                }
            }
        }

        /* Mirror the upper bands into the lower bands */
        for (m = 1; m <= 2 * delta_x; m++)
            for (x = 0; x < ncols - m; x++)
                p_Aij[paij_index(x + m, 2 * delta_x - m)] = p_Aij[paij_index(x, 2 * delta_x + m)];

        if (lambda_sP > 0.e0)
        {
            lambda = lambda_sP;

            p_Aij[paij_index(0, 2 * delta_x)] += lambda;     /* Main diagonal  */
            p_Aij[paij_index(0, 2 * delta_x + 1)] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols - 1; x++)
            {
                p_Aij[paij_index(x, 2 * delta_x - 1)] -= lambda;    /* Lower diagonal */
                p_Aij[paij_index(x, 2 * delta_x)] += lambda * 2.e0; /* Main diagonal  */
                p_Aij[paij_index(x, 2 * delta_x + 1)] -= lambda;    /* Upper diagonal */
            }
            p_Aij[paij_index(ncols - 1, 2 * delta_x - 1)] -= lambda; /* Lower diagonal */
            p_Aij[paij_index(ncols - 1, 2 * delta_x)] += lambda;     /* Main diagonal  */
        }

#if REGULARIZE_DIAGONAL
        /* Regularize diagonal to prevent singular matrix from fully masked columns.
           When a column has no valid data (all pixels masked), the corresponding
           row of the matrix is zero, causing division by zero in bandsol.
           We add a small regularization to the diagonal to make it non-singular.
           The resulting spectrum value for masked columns will be ~0 (from p_bj[x]/diag). */
        {
            double max_diag = 0.0;
            for (x = 0; x < ncols; x++)
            {
                if (p_Aij[paij_index(x, 2 * delta_x)] > max_diag)
                    max_diag = p_Aij[paij_index(x, 2 * delta_x)];
            }
            if (max_diag > 0.0)
            {
                double min_diag = max_diag * 1.0e-10;
                for (x = 0; x < ncols; x++)
                {
                    if (p_Aij[paij_index(x, 2 * delta_x)] < min_diag)
                        p_Aij[paij_index(x, 2 * delta_x)] = min_diag;
                }
            }
        }
#endif

        /* Solve the system of equations */
        bandsol(p_Aij, p_bj, MAX_PAIJ_X, MAX_PAIJ_Y);

        /* Save old spectrum, update, and compute change */
        for (x = 0; x < ncols; x++)
            sP_old[x] = sP[sp_index(x)];
        for (x = 0; x < ncols; x++)
            sP[sp_index(x)] = p_bj[pbj_index(x)];
        for (x = 0; x < ncols; x++)
            sP_diff[x] = fabs(sP[sp_index(x)] - sP_old[x]);

        /* Convergence: 99th percentile of change relative to median spectrum */
        sP_change = quick_select_percentile(sP_diff, ncols, 99.0);
        for (x = 0; x < ncols; x++)
            sP_diff[x] = sP[sp_index(x)];  /* reuse buffer for median calc */
        sP_med = fabs(quick_select_median(sP_diff, ncols));

        /* Compute the model.
           y is the outer loop so that the zeta tensor (y-major, by far the
           largest array) is read sequentially, and model is written
           sequentially */
        if (fast_flat && nthreads > 1)
        {
            pc.sP = sP;
            pc.sL = sL;
            pc.k0col = k0col;
            pc.d1col = d1col;
            pc.model = model;
            pool_run(pool, flat_model_chunk, &pc, PAR_CHUNKS);
        }
        else if (fast_flat)
            flat_model(ncols, 0, nrows, osample, sP, sL, k0col, d1col,
                       flat_scratch, model);
        else if (nthreads > 1)
        {
            pc.sP = sP;
            pc.sL = sL;
            pc.zeta = zeta;
            pc.m_zeta = m_zeta;
            pc.model = model;
            pool_run(pool, model_chunk, &pc, PAR_CHUNKS);
        }
        else
        for (y = 0; y < nrows; y++)
        {
            for (x = 0; x < ncols; x++)
            {
                const zeta_ref *zrow = &zeta[zeta_index(x, y, 0)];
                const int mz = m_zeta[mzeta_index(x, y)];
                double acc = 0.;
                for (m = 0; m < mz; m++)
                {
                    xx = zrow[m].x;
                    iy = zrow[m].iy;
                    ww = zrow[m].w;
                    acc += sP[xx] * sL[iy] * ww;
                }
                model[im_index(x, y)] = acc;
            }
        }

        /* With a preset sL and no outlier rejection the sP solve does not
           depend on the previous iteration: one pass is exact. sP_change
           still holds the jump from the initial guess, which says nothing
           about convergence here, so report 0. */
        if (use_preset && kappa <= 0)
        {
            sP_change = 0;
            iter = 1;
            break;
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
        if (nthreads > 1)
        {
            pc.mask = mask;
            pc.model = model;
            pool_run(pool, dev_chunk, &pc, PAR_CHUNKS);
            tmp = 0;
            isum = 0;
            for (m = 0; m < PAR_CHUNKS; m++)
            {
                tmp += pc.devsum[m];
                isum += pc.devcnt[m];
            }
        }
        else
        {
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
                        isum++;
                    }
                }
            }
        }
        dev = sqrt(tmp / isum);

        /* Adjust the mask marking outliers */
        if (kappa > 0)
        {
            if (nthreads > 1)
            {
                pc.model = model;
                pc.thresh = kappa * dev;
                pc.mask_out = mask;
                pool_run(pool, maskadj_chunk, &pc, PAR_CHUNKS);
            }
            else
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
        printf("Iteration: %i, sP_change: %g, sP_med: %g, threshold: %g\n",
               iter, sP_change, sP_med, sP_stop * sP_med);
#endif
        /* Check for convergence: stop when the 99th-percentile spectrum change
           drops below sP_stop * median(sP). Always do at least 2 iterations;
           maxiter is an unconditional upper bound. */
    } while ((iter++ == 0) || ((iter <= maxiter) && (sP_change > sP_stop * sP_med)));

    /* The loop exits with iter == maxiter + 1 only when the convergence test
       never passed; converging exactly at iter == maxiter is a success. */
    if (iter > maxiter)
    {
        status = -1; // ran out of iterations
        success = 0;
    }
    else
        status = 1; // converged

    /* A non-finite spectrum (e.g. unmasked NaN pixels with kappa == 0) must
       not be reported as success. The convergence test cannot catch it: NaN
       compares false and exits the loop as if converged. */
    if (!isfinite(sP_change))
    {
        status = -3; // non-finite result
        success = 0;
    }

    /* Uncertainty estimate */

    for (x = 0; x < ncols; x++)
    {
        unc[sp_index(x)] = 0.;
        p_bj[pbj_index(x)] = 0.;
        p_Aij[paij_index(x, 0)] = 0;
    }

    if (fast_flat)
    {
        /* Summed (and summed squared) subpixel weights per column:
           accumulated in the same subpixel order as the zeta list */
        const double s = 1.e0 / osample;
        for (x = 0; x < ncols; x++)
        {
            double sw = d1col[x], sw2 = d1col[x] * d1col[x];
            for (m = 1; m < osample; m++)
            {
                sw += s;
                sw2 += s * s;
            }
            sw += s - d1col[x];
            sw2 += (s - d1col[x]) * (s - d1col[x]);
            swcol[x] = sw;
            sw2col[x] = sw2;
        }
        flat_unc(ncols, nrows, nx, im, model, mask, swcol, sw2col,
                 unc, p_bj, p_Aij);
    }
    else if (nthreads > 1)
    {
        int uk;
        pc.mask = mask;
        pc.model = model;
        atomic_store(&pc.fail, 0);
        pool_run(pool, unc_chunk, &pc, PAR_CHUNKS);
        if (atomic_load(&pc.fail))
        {
            for (uk = 0; uk < PAR_CHUNKS; uk++)
            {
                free(pc.uslice[uk]);
                pc.uslice[uk] = NULL;
            }
            goto unc_sequential;
        }
        for (uk = 0; uk < PAR_CHUNKS; uk++)
        {
            if (!pc.uslice[uk])
                continue;
            const double *U = pc.uslice[uk];
            for (x = 0; x < ncols; x++)
            {
                unc[sp_index(x)] += U[x];
                p_bj[pbj_index(x)] += U[ncols + x];
                p_Aij[paij_index(x, 0)] += U[2 * ncols + x];
            }
            free(pc.uslice[uk]);
            pc.uslice[uk] = NULL;
        }
    }
    else
    unc_sequential:
    /* y is the outer loop so zeta (y-major) is read sequentially */
    for (y = 0; y < nrows; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            if (!mask[im_index(x, y)])
                continue;
            const zeta_ref *zrow = &zeta[zeta_index(x, y, 0)];
            const int mz = m_zeta[mzeta_index(x, y)];
            // Should pix_unc contribute here?
            tmp = im[im_index(x, y)] - model[im_index(x, y)];
            for (m = 0; m < mz; m++)
            {
                xx = zrow[m].x;
                ww = zrow[m].w;
                unc[sp_index(xx)] += tmp * tmp * ww;
                p_bj[pbj_index(xx)] += ww;           // Norm
                p_Aij[paij_index(xx, 0)] += ww * ww; // Norm squared
            }
        }
    }

    for (x = 0; x < ncols; x++)
    {
        norm = p_bj[pbj_index(x)] - p_Aij[paij_index(x, 0)] / p_bj[pbj_index(x)];
        unc[sp_index(x)] = sqrt(unc[sp_index(x)] / norm * nrows);
    }

    for (x = 0; x < delta_x; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }
    for (x = ncols - delta_x; x < ncols; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }

    /* Scanned after the edge zeroing so the flag reflects the returned data */
    for (x = 0; x < ncols; x++)
    {
        if (!isfinite(sP[sp_index(x)]))
        {
            status = -3; // non-finite result
            success = 0;
            break;
        }
    }

    free(sP_old);
    free(sP_diff);
    free(l_Aij);
    free(p_Aij);
    free(p_bj);
    free(l_bj);
    free(zw);
    free(zk);
    pool_stop(pool);
    free(pc.zwall);
    free(pc.zkall);
    free(pc.fscratch);

    free(zeta);
    free(m_zeta);
    free(z_rng);
    free(ycen_offset);
    free(k0col);
    free(d1col);
    free(swcol);
    free(sw2col);
    free(flat_scratch);

    info[0] = success;
    info[1] = sP_change;
    info[2] = status;
    info[3] = iter;
    info[4] = delta_x;

    return 0;
}
