#!/usr/bin/env python
"""
Test the frequency-dependent smoothing function.

By default, exercises the function on a synthetic red-noise-like
wavenumber-frequency spectrum so the test has no external data
dependency. Pass --input to run against a real spectrum file
(a NetCDF with 'symmetric'/'antisymmetric' variables on
wavenumber/frequency dims, as produced by wavenumber_frequency_functions).
"""

import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
import wavenumber_frequency_functions as wf


def synthetic_spectrum(n_wavenumber=289, n_frequency=385):
    wavenumber = np.arange(n_wavenumber) - n_wavenumber // 2
    frequency = np.linspace(-2.0, 2.0, n_frequency)
    f0 = 0.15
    power = 1.0 / (1.0 + (frequency[None, :] / f0) ** 2)
    power = power * (1.0 + 0.05 * np.cos(wavenumber[:, None] * np.pi / n_wavenumber))
    rng = np.random.default_rng(0)
    noise = rng.gamma(shape=2.0, scale=0.5, size=power.shape)
    data = power * noise
    da = xr.DataArray(
        data,
        dims=("wavenumber", "frequency"),
        coords={"wavenumber": wavenumber, "frequency": frequency},
        name="symmetric",
    )
    return xr.Dataset({"symmetric": da, "antisymmetric": da})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None,
                         help="NetCDF spectrum file with symmetric/antisymmetric "
                              "variables. If omitted, uses a synthetic spectrum.")
    parser.add_argument("--output-dir", default=".",
                         help="Directory to write the comparison plot to.")
    args = parser.parse_args()

    print("=" * 70)
    print("TESTING FREQUENCY-DEPENDENT SMOOTHING FUNCTION")
    print("=" * 70)

    print("\n1. Loading spectrum...")
    if args.input:
        c_ds = xr.open_dataset(args.input)
        print(f"   Loaded from {args.input}")
    else:
        c_ds = synthetic_spectrum()
        print("   Using synthetic red-noise-like spectrum")
    print(f"   Dataset dims: {c_ds.dims}")

    print("\n2. Preparing test data...")
    test_data = c_ds['symmetric']
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Frequency range: {test_data.frequency.min().values:.4f} to {test_data.frequency.max().values:.4f} cpd")

    z2 = c_ds[['symmetric', 'antisymmetric']].to_array(dim='component')
    z2avg = z2.mean(dim='component')
    print(f"   Averaged spectrum shape: {z2avg.shape}")

    print("\n3. Applying frequency-dependent smoothing (default settings)...")
    print("   Frequency bands: [0.0, 0.2, 0.5, 2.0] cpd")
    print("   Smoothing passes: [100, 50, 20]")

    background_freqdep = wf.smooth_wavefreq_frequency_dependent(
        z2avg,
        freq_name='frequency'
    )
    print("   ✓ Frequency-dependent smoothing complete")

    print("\n4. Applying original uniform smoothing (50 passes)...")
    background_original = wf.smooth_wavefreq(
        z2avg,
        kern=wf.simple_smooth_kernel(),
        nsmooth=50,
        freq_name='frequency'
    )
    print("   ✓ Original smoothing complete")

    print("\n5. Comparing smoothing methods...")
    print("=" * 70)

    zero_freq_idx = np.argmin(np.abs(c_ds.frequency.values))
    low_freq_indices = list(range(zero_freq_idx + 1, zero_freq_idx + 11))

    print("\nBackground values at low frequencies (wavenumber=0):")
    print("Freq (cpd)  | Original BG | FreqDep BG | Ratio (FD/Orig)")
    print("-" * 70)

    for idx in low_freq_indices[:5]:
        freq_val = c_ds.frequency.values[idx]
        bg_orig = background_original.sel(wavenumber=0).isel(frequency=idx).values
        bg_freqdep = background_freqdep.sel(wavenumber=0).isel(frequency=idx).values
        ratio = bg_freqdep / bg_orig if bg_orig > 0 else np.nan

        print(f"{freq_val:.5f}   | {bg_orig:11.4f} | {bg_freqdep:10.4f} | {ratio:6.3f}")

    print("\n6. Calculating normalized spectra...")
    z2_sym = c_ds['symmetric']
    z2_asy = c_ds['antisymmetric']

    norm_sym_orig = z2_sym / background_original
    norm_asy_orig = z2_asy / background_original

    norm_sym_freqdep = z2_sym / background_freqdep
    norm_asy_freqdep = z2_asy / background_freqdep

    print("\nNormalized power at low frequencies (mean over wavenumber -15 to 15):")
    print("Freq (cpd)  | Original Sym | FreqDep Sym | Original Asym | FreqDep Asym")
    print("-" * 70)

    for idx in low_freq_indices[:5]:
        freq_val = c_ds.frequency.values[idx]

        orig_sym = norm_sym_orig.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        fd_sym = norm_sym_freqdep.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        orig_asy = norm_asy_orig.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        fd_asy = norm_asy_freqdep.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values

        print(f"{freq_val:.5f}   | {orig_sym:12.3f} | {fd_sym:11.3f} | {orig_asy:13.3f} | {fd_asy:12.3f}")

    print("\n" + "=" * 70)
    print("Expected improvement:")
    print("  - Frequency-dependent method should have normalized power closer to 1.0")
    print("  - Especially at the lowest frequencies (0.01-0.05 cpd)")
    print("=" * 70)

    print("\n7. Generating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    freq_vals = c_ds.frequency.isel(frequency=low_freq_indices).values
    bg_orig_wn0 = background_original.sel(wavenumber=0).isel(frequency=low_freq_indices).values
    bg_fd_wn0 = background_freqdep.sel(wavenumber=0).isel(frequency=low_freq_indices).values
    raw_sym_wn0 = z2_sym.sel(wavenumber=0).isel(frequency=low_freq_indices).values

    ax.plot(freq_vals, raw_sym_wn0, 'k-', label='Raw Symmetric', linewidth=2, marker='o')
    ax.plot(freq_vals, bg_orig_wn0, 'b--', label='Original BG (50 passes)', linewidth=2, marker='s')
    ax.plot(freq_vals, bg_fd_wn0, 'r-', label='FreqDep BG (100/50/20)', linewidth=2, marker='^')
    ax.set_xlabel('Frequency (cpd)', fontsize=11)
    ax.set_ylabel('Power', fontsize=11)
    ax.set_title('Background Comparison (Wavenumber=0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i in range(5):
        idx = low_freq_indices[i]
        freq_val = c_ds.frequency.values[idx]
        norm_orig = norm_sym_orig.isel(frequency=idx).sel(wavenumber=slice(-15, 15))
        norm_fd = norm_sym_freqdep.isel(frequency=idx).sel(wavenumber=slice(-15, 15))

        ax.plot(norm_orig.wavenumber, norm_orig, 'o-', label=f'Orig f={freq_val:.3f}', alpha=0.6)
        ax.plot(norm_fd.wavenumber, norm_fd, 's--', label=f'FD f={freq_val:.3f}', alpha=0.6)

    ax.axhline(1.0, color='k', linestyle='-', linewidth=2, label='Expected (1.0)')
    ax.set_xlabel('Wavenumber', fontsize=11)
    ax.set_ylabel('Normalized Power', fontsize=11)
    ax.set_title('Symmetric: Original vs FreqDep', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3])

    ax = axes[1, 0]
    freq_vals = [c_ds.frequency.values[idx] for idx in low_freq_indices]
    mean_orig_sym = [norm_sym_orig.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                     for idx in low_freq_indices]
    mean_fd_sym = [norm_sym_freqdep.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                   for idx in low_freq_indices]

    ax.plot(freq_vals, mean_orig_sym, 'bo-', label='Original (Sym)', linewidth=2, markersize=8)
    ax.plot(freq_vals, mean_fd_sym, 'rs-', label='FreqDep (Sym)', linewidth=2, markersize=8)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2, label='Expected (1.0)')
    ax.set_xlabel('Frequency (cpd)', fontsize=11)
    ax.set_ylabel('Mean Normalized Power', fontsize=11)
    ax.set_title('Mean Normalized Power: Original vs FreqDep', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2.5])

    ax = axes[1, 1]
    std_orig_sym = [norm_sym_orig.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).std().values
                    for idx in low_freq_indices]
    std_fd_sym = [norm_sym_freqdep.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).std().values
                  for idx in low_freq_indices]

    ax.plot(freq_vals, std_orig_sym, 'bo-', label='Original (Sym)', linewidth=2, markersize=8)
    ax.plot(freq_vals, std_fd_sym, 'rs-', label='FreqDep (Sym)', linewidth=2, markersize=8)
    ax.set_xlabel('Frequency (cpd)', fontsize=11)
    ax.set_ylabel('Std Dev of Normalized Power', fontsize=11)
    ax.set_title('Variability: Original vs FreqDep', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{args.output_dir}/test_frequency_dependent_smoothing.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Comparison plot saved: {out_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("  1. Function executed without errors ✓")
    print("  2. Background values computed for both methods ✓")
    print("  3. Normalized spectra calculated ✓")
    print("  4. Comparison plot generated ✓")
    print("\nNext steps:")
    print("  - Review the comparison plot to assess improvement")
    print("  - If successful, apply to all three datasets")
    print("  - Regenerate Wheeler-Kiladis diagrams with new background")


if __name__ == "__main__":
    main()
