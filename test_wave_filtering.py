"""Correctness checks for wave_filtering.py.

Each check prints one ``PASS/FAIL metric=…`` line. Runs on the bundled
``OLR.12hr_2yrs.wheeler.nc`` test dataset (730 timesteps of 12-hourly OLR, spd=2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wave_filtering import (
    RIOS_BERRIOS_FILTERS,
    apply_mask_ifft,
    build_box_mask,
    fft_spacetime,
    filter_wave,
    prepare_for_filter,
    trim_padding,
)

TEST_NC = HERE / "OLR.12hr_2yrs.wheeler.nc"
SPD = 2.0  # OLR test file is 12-hourly
PAD = 700
CHECKPOINT_DIR = HERE.parent / "cam_wk_spectra" / "checkpoints"

PASS = 0
FAIL = 0


def report(name: str, ok: bool, metric: str) -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{tag}] {name}: {metric}")


def load_olr_merid(last_n: int | None = None) -> xr.DataArray:
    ds = xr.open_dataset(TEST_NC, decode_times=False)
    da = ds["olr"].sel(lat=slice(-10, 10)).mean("lat")
    if last_n is not None:
        da = da.isel(time=slice(-last_n, None))
    return da


# ─── Check 1: round-trip identity ────────────────────────────────────────────
def check_round_trip():
    da = load_olr_merid()
    padded, n_orig = prepare_for_filter(da, pad_zeros=PAD, detrend=False, remove_mean=True)
    z = fft_spacetime(padded, spd=SPD)
    # mask all-ones → ifft(fft(x)) == x
    mask = xr.DataArray(
        np.ones((z.sizes["frequency"], z.sizes["wavenumber"])),
        dims=("frequency", "wavenumber"),
        coords={"frequency": z.frequency, "wavenumber": z.wavenumber},
    )
    recovered = apply_mask_ifft(z, mask)
    rel = float(np.max(np.abs(recovered.values - padded.values))
                / max(np.ptp(padded.values), 1e-30))
    report("round_trip_identity", rel < 1e-10, f"max_rel_err={rel:.2e}")


# ─── Check 2: Parseval on the padded filtered field ──────────────────────────
def check_parseval():
    da = load_olr_merid()
    padded, n_orig = prepare_for_filter(da, pad_zeros=PAD, remove_mean=True)
    z = fft_spacetime(padded, spd=SPD)
    nt_pad = z.sizes["frequency"]
    nx = z.sizes["wavenumber"]

    worst = 0.0
    for wt, spec in RIOS_BERRIOS_FILTERS.items():
        m = build_box_mask(z.wavenumber, z.frequency, **spec)
        y = apply_mask_ifft(z, m)
        energy_space = float(np.sum(y.values ** 2))
        spec_sq = np.abs(z.values * m.values) ** 2
        energy_spec = float(spec_sq.sum() / (nt_pad * nx))
        rel = abs(energy_space - energy_spec) / max(energy_space, 1e-30)
        worst = max(worst, rel)
    report("parseval_consistency", worst < 1e-8, f"max_rel_err={worst:.2e}")


# ─── Check 3: linearity ──────────────────────────────────────────────────────
def check_linearity():
    da = load_olr_merid()
    # Build two independent fields of the same shape.
    rng = np.random.default_rng(0)
    a = da.copy(deep=True)
    b = da.copy(deep=True)
    b.values = rng.standard_normal(b.shape) * float(a.std())

    worst = 0.0
    for wt in RIOS_BERRIOS_FILTERS:
        fa = filter_wave(a, wt, spd=SPD, pad_zeros=PAD)
        fb = filter_wave(b, wt, spd=SPD, pad_zeros=PAD)
        fab = filter_wave(a + b, wt, spd=SPD, pad_zeros=PAD)
        diff = float(np.max(np.abs(fab.values - (fa.values + fb.values))))
        ref = float(np.ptp(fab.values))
        rel = diff / max(ref, 1e-30)
        worst = max(worst, rel)
    report("linearity", worst < 1e-10, f"max_rel_err={worst:.2e}")


# ─── Check 4: mask-all identity (mask=1 except the single DC mode) ───────────
def check_mask_all():
    da = load_olr_merid()
    padded, n_orig = prepare_for_filter(da, pad_zeros=PAD, remove_mean=True)
    z = fft_spacetime(padded, spd=SPD)
    nf = z.sizes["frequency"]
    nk = z.sizes["wavenumber"]
    mask_vals = np.ones((nf, nk))
    # Zero ONLY the single DC mode (f=0 AND k=0), not the whole f=0 row or k=0 column.
    # Removing k=0 would drop the zonal-mean time series (nonzero even after mean-removal).
    f_arr = z.frequency.values
    k_arr = z.wavenumber.values
    i_f0 = int(np.where(f_arr == 0)[0][0])
    i_k0 = int(np.where(k_arr == 0)[0][0])
    mask_vals[i_f0, i_k0] = 0.0
    mask = xr.DataArray(mask_vals, dims=("frequency", "wavenumber"),
                        coords={"frequency": f_arr, "wavenumber": k_arr})
    y = apply_mask_ifft(z, mask)
    diff = float(np.max(np.abs(y.values - padded.values)))
    rel = diff / max(np.ptp(padded.values), 1e-30)
    report("mask_all_identity", rel < 1e-10, f"max_rel_err={rel:.2e}")


# ─── Check 5: subset/energy-containment bounds ───────────────────────────────
def check_band_bounds():
    """Each wave band's variance must be ≤ union variance ≤ total variance.

    Note: the Rios-Berrios advective and WIG boxes overlap in (westward, k=4-14,
    T=2.5-4.5), so sum(var_wt) > var_union in general. The invariant that
    definitely holds is the subset property: each mask ⊂ union, so each filtered
    field has variance ≤ the union-filtered variance, which in turn ≤ total.
    """
    da = load_olr_merid()
    padded, n_orig = prepare_for_filter(da, pad_zeros=PAD, remove_mean=True)
    z = fft_spacetime(padded, spd=SPD)

    masks = {wt: build_box_mask(z.wavenumber, z.frequency, **spec)
             for wt, spec in RIOS_BERRIOS_FILTERS.items()}
    overlap = sum(m.values for m in masks.values())

    vars_ = {}
    for wt, m in masks.items():
        y = apply_mask_ifft(z, m)
        y_trim = trim_padding(y, n_orig, reference_time=da.time, reference_lon=da.lon)
        vars_[wt] = float((y_trim ** 2).mean())

    union_mask = xr.DataArray(
        (overlap > 0).astype(float), dims=("frequency", "wavenumber"),
        coords={"frequency": z.frequency, "wavenumber": z.wavenumber},
    )
    y_u = apply_mask_ifft(z, union_mask)
    y_u_trim = trim_padding(y_u, n_orig, reference_time=da.time, reference_lon=da.lon)
    var_union = float((y_u_trim ** 2).mean())
    merid_anom = da - da.mean("time")
    var_total = float((merid_anom ** 2).mean())

    subset_ok = all(v <= var_union * (1 + 1e-10) for v in vars_.values())
    containment_ok = var_union <= var_total * (1 + 1e-10)
    ok = subset_ok and containment_ok
    worst_wt = max(vars_, key=vars_.get)
    report(
        "band_bounds",
        ok,
        f"max_var_wt={vars_[worst_wt]:.3f} ({worst_wt}) "
        f"<= var_union={var_union:.3f} <= var_total={var_total:.3f}",
    )


# ─── Check 6: cross-package against tropical_diagnostics.spacetime.kf_filter ─
def check_cross_package():
    try:
        from tropical_diagnostics.spacetime import kf_filter
    except Exception as e:
        report("cross_package_tropdiag", False, f"import_error={e}")
        return

    # 100-day chunk for speed. pad_zeros=0 so both codes operate on the raw block.
    nday = 100
    ntime = int(nday * SPD)
    da = load_olr_merid(last_n=ntime)

    # Ours — Kelvin with no zero-padding
    k_ours = filter_wave(da, "kelvin", spd=SPD, pad_zeros=0, remove_mean=True)
    v_ours = float((k_ours ** 2).mean())

    # Theirs — same k/period bounds, hMin=hMax=-9999 disables dispersion bounds.
    arr = (da - da.mean("time")).values
    k_theirs_arr = kf_filter(
        arr, obsPerDay=int(SPD),
        tMin=2.5, tMax=20.0, kMin=1, kMax=14,
        hMin=-9999, hMax=-9999, waveName="Kelvin",
    )
    v_theirs = float(np.mean(k_theirs_arr ** 2))

    rel = abs(v_ours - v_theirs) / max(v_theirs, 1e-30)
    # 5% tolerance accounts for kf_filter's rfft2+round-at-boundaries vs our
    # fft2+exact-comparison; the two should agree on interior energy tightly
    # but can disagree on which k/f row lands on the mask edge.
    report("cross_package_tropdiag", rel < 0.05,
           f"ours={v_ours:.3f} theirs={v_theirs:.3f} rel_diff={rel:.2%}")


# ─── Check 7: Kelvin phase-speed sanity ──────────────────────────────────────
def check_kelvin_phase_speed():
    # Use a last-100-day block to mimic the paper setup.
    nday = 100
    ntime = int(nday * SPD)
    da = load_olr_merid(last_n=ntime)
    k_filt = filter_wave(da, "kelvin", spd=SPD, pad_zeros=PAD, remove_mean=True)

    # Lag-longitude cross-correlation of the zonal wave pattern.
    # Take the longitude time series at lon=0 reference, correlate against all
    # longitudes at a +1-step time lag; the longitude of maximum correlation gives
    # the distance the wave travelled in one 12-hourly step.
    arr = k_filt.values  # (time, lon)
    nt, nx = arr.shape
    ref = arr[:-1, 0]
    lags = np.zeros(nx)
    ref_std = ref.std()
    for i in range(nx):
        tgt = arr[1:, i]
        s = ref_std * tgt.std()
        lags[i] = (ref * tgt).mean() / s if s > 0 else 0.0

    best_i = int(np.argmax(lags))
    # Convert degrees-per-12h to m/s at the equator.
    lon = da.lon.values
    dlon_deg = lon[best_i] - lon[0] if best_i <= nx // 2 else (lon[best_i] - lon[0]) - 360.0
    dt_sec = (24 * 3600) / SPD  # 12 h = 43200 s
    R_eq = 6.371e6
    c = (dlon_deg * np.pi / 180.0) * R_eq / dt_sec
    ok = 5.0 <= c <= 35.0
    report("kelvin_phase_speed", ok, f"c={c:.1f} m/s (expect ~10-20)")


# ─── Check 8: mask-overlay plot ──────────────────────────────────────────────
def check_mask_overlay_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        report("mask_overlay_plot", False, f"matplotlib_import_error={e}")
        return

    da = load_olr_merid()
    padded, _ = prepare_for_filter(da, pad_zeros=PAD, remove_mean=True)
    z = fft_spacetime(padded, spd=SPD)

    # Display on the "plottable" subgrid: positive frequencies, all wavenumbers.
    f = z.frequency.values
    k = z.wavenumber.values
    f_pos = f > 0
    k_sort = np.argsort(k)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for wt, spec in RIOS_BERRIOS_FILTERS.items():
        m = build_box_mask(z.wavenumber, z.frequency, **spec).values
        m_plot = m[np.ix_(f_pos, k_sort)]
        power = np.abs(z.values[np.ix_(f_pos, k_sort)]) ** 2
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.pcolormesh(
            k[k_sort], f[f_pos], np.log10(power + 1e-20),
            cmap="Greys", shading="auto",
        )
        ax.contour(k[k_sort], f[f_pos], m_plot, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("zonal wavenumber")
        ax.set_ylabel("frequency (cycles / day)")
        ax.set_title(f"{wt}: k={spec['k_range']} T={spec['period_range']} d  {spec['direction']}")
        fig.colorbar(im, ax=ax, label="log10 |FFT|²")
        out = CHECKPOINT_DIR / f"mask_overlay_{wt}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
    report("mask_overlay_plot", True, f"saved 3 PNGs to {CHECKPOINT_DIR}")


def main():
    checks = [
        check_round_trip,
        check_parseval,
        check_linearity,
        check_mask_all,
        check_band_bounds,
        check_cross_package,
        check_kelvin_phase_speed,
        check_mask_overlay_plot,
    ]
    print(f"=== wave_filtering correctness checks (input: {TEST_NC.name}) ===\n")
    for c in checks:
        try:
            c()
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"[FAIL] {c.__name__}: exception={type(e).__name__}: {e}")
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
