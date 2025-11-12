"""
Direct speed comparison between Python and C implementations.
"""

import numpy as np
import time
from extract_python import extract as extract_python
from test_extract_python import create_test_data

try:
    import charslit
    HAS_C = True
except ImportError:
    HAS_C = False
    print("C version not available")
    exit(1)


def extract_c_wrapper(ncols, nrows, im, pix_unc, mask, ycen, slitdeltas,
                     osample, lambda_sP, lambda_sL, maxiter, slit_func_in=None):
    """Wrapper for C implementation."""
    result = charslit.extract(
        im, pix_unc, mask.astype(np.int32), ycen, slitdeltas,
        osample, lambda_sP, lambda_sL, maxiter, slit_func_in
    )
    _, sL, sP, model, unc, info, _, _ = result
    return sP, sL, model, unc, info


def benchmark_size(ncols, nrows, osample=5, runs=3):
    """Benchmark both implementations for a given size."""
    print(f"\n{'='*70}")
    print(f"Benchmarking {ncols}×{nrows} pixels (osample={osample})")
    print(f"{'='*70}")

    # Create test data
    data = create_test_data(ncols, nrows, osample, noise_level=0.1)

    # Warmup
    _ = extract_python(ncols, nrows, data['im'], data['pix_unc'], data['mask'],
                      data['ycen'], data['slitdeltas'], osample, 0.1, 0.1, 20)

    _ = extract_c_wrapper(ncols, nrows, data['im'], data['pix_unc'], data['mask'],
                         data['ycen'], data['slitdeltas'], osample, 0.1, 0.1, 20)

    # Python benchmark
    py_times = []
    for i in range(runs):
        start = time.time()
        sP_py, sL_py, model_py, unc_py, info_py = extract_python(
            ncols, nrows, data['im'], data['pix_unc'], data['mask'],
            data['ycen'], data['slitdeltas'], osample, 0.1, 0.1, 20
        )
        py_time = time.time() - start
        py_times.append(py_time)
        print(f"  Python run {i+1}: {py_time:.4f}s (cost: {info_py[1]:.2f})")

    py_avg = np.mean(py_times)
    py_std = np.std(py_times)

    # C benchmark
    c_times = []
    for i in range(runs):
        start = time.time()
        sP_c, sL_c, model_c, unc_c, info_c = extract_c_wrapper(
            ncols, nrows, data['im'], data['pix_unc'], data['mask'],
            data['ycen'], data['slitdeltas'], osample, 0.1, 0.1, 20
        )
        c_time = time.time() - start
        c_times.append(c_time)
        print(f"  C run {i+1}:      {c_time:.4f}s (cost: {info_c[1]:.2f})")

    c_avg = np.mean(c_times)
    c_std = np.std(c_times)

    # Summary
    speedup = py_avg / c_avg
    total_pixels = ncols * nrows

    print(f"\n  Summary:")
    print(f"    Total pixels: {total_pixels}")
    print(f"    Python: {py_avg:.4f}s ± {py_std:.4f}s  ({py_avg*1000/total_pixels:.2f} ms/pixel)")
    print(f"    C:      {c_avg:.4f}s ± {c_std:.4f}s  ({c_avg*1000/total_pixels:.2f} ms/pixel)")
    print(f"    Speedup: {speedup:.1f}x")

    return {
        'size': f"{ncols}×{nrows}",
        'pixels': total_pixels,
        'py_time': py_avg,
        'c_time': c_avg,
        'speedup': speedup
    }


def main():
    """Run benchmarks for multiple sizes."""
    print("\n" + "="*70)
    print("PYTHON vs C SPEED COMPARISON")
    print("="*70)

    # Test different sizes
    sizes = [
        (30, 15, 5),    # Small
        (50, 20, 5),    # Medium
        (100, 40, 5),   # Large
        (200, 60, 5),   # Extra large
    ]

    results = []
    for ncols, nrows, osample in sizes:
        try:
            result = benchmark_size(ncols, nrows, osample, runs=3)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Final summary table
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':<12s} {'Pixels':>8s} {'Python':>10s} {'C':>10s} {'Speedup':>10s}")
    print("-"*70)

    for r in results:
        print(f"{r['size']:<12s} {r['pixels']:>8d} "
              f"{r['py_time']:>9.4f}s {r['c_time']:>9.4f}s "
              f"{r['speedup']:>9.1f}x")

    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        print(f"C version is approximately {int(avg_speedup)}× faster than Python")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
