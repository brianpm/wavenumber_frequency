#!/usr/bin/env python
"""
Test parametric background estimation methods.

By default, exercises the methods on a synthetic red-noise-like
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
    print("TESTING PARAMETRIC BACKGROUND ESTIMATION")
    print("=" * 70)

    print("\n1. Loading spectrum...")
    if args.input:
        c_ds = xr.open_dataset(args.input)
        print(f"   Loaded from {args.input}")
    else:
        c_ds = synthetic_spectrum()
        print("   Using synthetic red-noise-like spectrum")

    print("\n2. Preparing averaged spectrum...")
    z2 = c_ds[['symmetric', 'antisymmetric']].to_array(dim='component')
    z2avg = z2.mean(dim='component')
    print(f"   Averaged spectrum shape: {z2avg.shape}")

    print("\n3. Computing backgrounds with different methods...")
    print("=" * 70)

    print("\n[1/4] Original uniform smoothing (50 passes)...")
    bg_original = wf.smooth_wavefreq(z2avg, nsmooth=50, freq_name='frequency')

    print("\n[2/4] Polynomial parametric model (degree 3)...")
    bg_polynomial = wf.estimate_background_parametric(
        z2avg,
        method='polynomial',
        poly_degree=3,
        initial_smooth_passes=20,
        freq_name='frequency'
    )

    print("\n[3/4] Red-noise parametric model...")
    bg_rednoise = wf.estimate_background_parametric(
        z2avg,
        method='rednoise',
        initial_smooth_passes=20,
        freq_name='frequency'
    )

    print("\n[4/4] Hybrid parametric model...")
    bg_hybrid = wf.estimate_background_parametric(
        z2avg,
        method='hybrid',
        poly_degree=3,
        initial_smooth_passes=20,
        freq_name='frequency'
    )

    print("\n✓ All background methods computed successfully")

    print("\n4. Comparing background estimates...")
    print("=" * 70)

    zero_freq_idx = np.argmin(np.abs(c_ds.frequency.values))
    low_freq_indices = list(range(zero_freq_idx + 1, zero_freq_idx + 11))

    print("\nBackground values at low frequencies (wavenumber=0):")
    print("Freq (cpd)  | Original | Polynomial | RedNoise | Hybrid")
    print("-" * 70)

    for idx in low_freq_indices[:5]:
        freq_val = c_ds.frequency.values[idx]
        bg_orig = bg_original.sel(wavenumber=0).isel(frequency=idx).values
        bg_poly = bg_polynomial.sel(wavenumber=0).isel(frequency=idx).values
        bg_red = bg_rednoise.sel(wavenumber=0).isel(frequency=idx).values
        bg_hyb = bg_hybrid.sel(wavenumber=0).isel(frequency=idx).values

        print(f"{freq_val:.5f}   | {bg_orig:8.4f} | {bg_poly:10.4f} | {bg_red:8.4f} | {bg_hyb:6.4f}")

    print("\n5. Calculating normalized spectra...")
    z2_sym = c_ds['symmetric']

    norm_original = z2_sym / bg_original
    norm_polynomial = z2_sym / bg_polynomial
    norm_rednoise = z2_sym / bg_rednoise
    norm_hybrid = z2_sym / bg_hybrid

    print("\nNormalized power (Symmetric, mean over wavenumber -15 to 15):")
    print("Freq (cpd)  | Original | Polynomial | RedNoise | Hybrid")
    print("-" * 70)

    for idx in low_freq_indices[:5]:
        freq_val = c_ds.frequency.values[idx]

        n_orig = norm_original.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        n_poly = norm_polynomial.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        n_red = norm_rednoise.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
        n_hyb = norm_hybrid.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values

        print(f"{freq_val:.5f}   | {n_orig:8.3f} | {n_poly:10.3f} | {n_red:8.3f} | {n_hyb:6.3f}")

    print("\n" + "=" * 70)
    print("Target: Normalized power should be close to 1.0")
    print("=" * 70)

    print("\n6. Generating comparison plots...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    freq_vals = c_ds.frequency.isel(frequency=low_freq_indices).values
    raw_sym = z2_sym.sel(wavenumber=0).isel(frequency=low_freq_indices).values

    ax1.plot(freq_vals, raw_sym, 'k-', label='Raw Symmetric', linewidth=2.5, marker='o', markersize=6)
    ax1.plot(freq_vals, bg_original.sel(wavenumber=0).isel(frequency=low_freq_indices).values,
             '--', label='Original (50 passes)', linewidth=2, marker='s')
    ax1.plot(freq_vals, bg_polynomial.sel(wavenumber=0).isel(frequency=low_freq_indices).values,
             '-', label='Polynomial (deg 3)', linewidth=2, marker='^')
    ax1.plot(freq_vals, bg_rednoise.sel(wavenumber=0).isel(frequency=low_freq_indices).values,
             '-', label='Red-noise', linewidth=2, marker='v')
    ax1.plot(freq_vals, bg_hybrid.sel(wavenumber=0).isel(frequency=low_freq_indices).values,
             '-', label='Hybrid', linewidth=2, marker='d')

    ax1.set_xlabel('Frequency (cpd)', fontsize=11)
    ax1.set_ylabel('Power', fontsize=11)
    ax1.set_title('Background Comparison (Wavenumber=0)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    freq_plot = [c_ds.frequency.values[idx] for idx in low_freq_indices]
    mean_orig = [norm_original.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                 for idx in low_freq_indices]
    mean_poly = [norm_polynomial.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                 for idx in low_freq_indices]
    mean_red = [norm_rednoise.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                for idx in low_freq_indices]
    mean_hyb = [norm_hybrid.isel(frequency=idx).sel(wavenumber=slice(-15, 15)).mean().values
                for idx in low_freq_indices]

    ax2.plot(freq_plot, mean_orig, 'o-', label='Original', linewidth=2, markersize=7)
    ax2.plot(freq_plot, mean_poly, 's-', label='Polynomial', linewidth=2, markersize=7)
    ax2.plot(freq_plot, mean_red, '^-', label='Red-noise', linewidth=2, markersize=7)
    ax2.plot(freq_plot, mean_hyb, 'd-', label='Hybrid', linewidth=2, markersize=7)
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=2, label='Expected (1.0)')

    ax2.set_xlabel('Frequency (cpd)', fontsize=11)
    ax2.set_ylabel('Mean Normalized Power', fontsize=11)
    ax2.set_title('Mean Normalized Power: Method Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 2.5])

    for plot_idx in range(4):
        ax = fig.add_subplot(gs[1 + plot_idx // 2, plot_idx % 2])
        idx = low_freq_indices[plot_idx]
        freq_val = c_ds.frequency.values[idx]

        wn_range = slice(-15, 15)
        ax.plot(norm_original.wavenumber.sel(wavenumber=wn_range),
                norm_original.isel(frequency=idx).sel(wavenumber=wn_range),
                'o-', label='Original', alpha=0.7, linewidth=1.5)
        ax.plot(norm_polynomial.wavenumber.sel(wavenumber=wn_range),
                norm_polynomial.isel(frequency=idx).sel(wavenumber=wn_range),
                's-', label='Polynomial', alpha=0.7, linewidth=1.5)
        ax.plot(norm_rednoise.wavenumber.sel(wavenumber=wn_range),
                norm_rednoise.isel(frequency=idx).sel(wavenumber=wn_range),
                '^-', label='Red-noise', alpha=0.7, linewidth=1.5)
        ax.plot(norm_hybrid.wavenumber.sel(wavenumber=wn_range),
                norm_hybrid.isel(frequency=idx).sel(wavenumber=wn_range),
                'd-', label='Hybrid', alpha=0.7, linewidth=1.5)

        ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Wavenumber', fontsize=10)
        ax.set_ylabel('Normalized Power', fontsize=10)
        ax.set_title(f'f = {freq_val:.4f} cpd', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 3])

    plt.suptitle('Parametric Background Model Comparison', fontsize=14, fontweight='bold')
    out_path = f"{args.output_dir}/test_parametric_background.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Comparison plot saved: {out_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nRecommendation:")
    print("  - Review the comparison plot to identify the best method")
    print("  - Look for method that brings normalized power closest to 1.0")
    print("  - Polynomial or Hybrid methods likely perform best for sharp low-freq features")


if __name__ == "__main__":
    main()
