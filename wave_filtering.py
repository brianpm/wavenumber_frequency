"""Wave-filtering pipeline: mask (k, f) spectrum, inverse-FFT to space-time.

Reproduces the filter used in Rios-Berrios et al. (2022, JAMES 14, e2021MS002902,
§2.2.2): meridional average → zero-pad in time → 2D FFT → zero outside a box in
(wavenumber, period) → inverse 2D FFT → trim padding. Rios-Berrios use simple
boxes in k-f space (no dispersion-curve bounds); a separate ``build_dispersion_mask``
helper wraps the existing Wheeler-Kiladis dispersion masks for users who want
tighter regions.

Sign convention (with numpy.fft.fft2 on axes (time, lon) of a real field):
  eastward wave exp(i(k*x - omega*t)) lives at (f<0, k>0) and its conjugate (f>0, k<0)
  westward wave exp(i(k*x + omega*t)) lives at (f>0, k>0) and its conjugate (f<0, k<0)
i.e. sign(k) * sign(f) < 0 → eastward,  sign(k) * sign(f) > 0 → westward.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.signal import detrend as _detrend


# Rios-Berrios et al. 2022 §2.2.2 — exact filter boxes
RIOS_BERRIOS_FILTERS: Mapping[str, Mapping] = {
    "kelvin":    dict(k_range=(1, 14),  period_range=(2.5, 20.0), direction="eastward"),
    "advective": dict(k_range=(4, 28),  period_range=(2.5, 20.0), direction="westward"),
    "wig_n1":    dict(k_range=(1, 14),  period_range=(1.8, 4.5),  direction="westward"),
}


def _axis_index(da: xr.DataArray, name: str) -> int:
    return da.dims.index(name)


def prepare_for_filter(
    data: xr.DataArray,
    pad_zeros: int = 700,
    detrend: bool = False,
    remove_mean: bool = True,
) -> tuple[xr.DataArray, int]:
    """Remove mean, optionally detrend, and zero-pad the time axis.

    Returns (padded_array, original_time_length). The padding is appended at the
    end of the time axis, matching Wheeler & Weickmann (2001) / Rios-Berrios (2022).
    """
    if "time" not in data.dims:
        raise ValueError(f"input must have a 'time' dim; got {data.dims}")

    time_axis = _axis_index(data, "time")
    n_time = data.sizes["time"]
    arr = data.values.astype(np.float64, copy=True)

    if detrend:
        arr = _detrend(arr, axis=time_axis, type="linear")
    if remove_mean:
        arr = arr - arr.mean(axis=time_axis, keepdims=True)

    if pad_zeros > 0:
        pad_shape = list(arr.shape)
        pad_shape[time_axis] = pad_zeros
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=time_axis)

    # Rebuild an xarray with a synthetic padded time coord (monotonic, keeps dt).
    if n_time >= 2:
        dt = float(data.time.values[1] - data.time.values[0])
    else:
        dt = 1.0
    new_time = np.arange(n_time + pad_zeros, dtype=np.float64) * dt + float(data.time.values[0])
    coords = {d: data.coords[d] for d in data.dims if d != "time"}
    coords["time"] = new_time
    padded = xr.DataArray(arr, dims=data.dims, coords=coords, name=data.name)
    padded.attrs.update(data.attrs)
    return padded, n_time


def fft_spacetime(data_padded: xr.DataArray, spd: float) -> xr.DataArray:
    """2D complex FFT over (time, lon); return complex DataArray with k, f coords.

    Wavenumber is integer zonal wavenumber (fftfreq * nlon).
    Frequency is cycles-per-day (fftfreq with d = 1/spd).
    """
    for d in ("time", "lon"):
        if d not in data_padded.dims:
            raise ValueError(f"input must have '{d}' dim; got {data_padded.dims}")
    t_ax = _axis_index(data_padded, "time")
    x_ax = _axis_index(data_padded, "lon")

    nt = data_padded.sizes["time"]
    nx = data_padded.sizes["lon"]

    z = np.fft.fft2(data_padded.values, axes=(t_ax, x_ax))

    freq = np.fft.fftfreq(nt, d=1.0 / spd)          # cycles per day
    wavenumber = np.fft.fftfreq(nx, d=1.0 / nx)     # integer wavenumbers
    wavenumber = np.rint(wavenumber).astype(np.int64)

    # Replace time/lon coords with frequency/wavenumber
    new_dims = tuple(("frequency" if d == "time" else "wavenumber" if d == "lon" else d)
                     for d in data_padded.dims)
    new_coords = {}
    for d, nd in zip(data_padded.dims, new_dims):
        if d == "time":
            new_coords["frequency"] = freq
        elif d == "lon":
            new_coords["wavenumber"] = wavenumber
        else:
            new_coords[d] = data_padded.coords[d]
    return xr.DataArray(z, dims=new_dims, coords=new_coords)


def build_box_mask(
    wavenumber: np.ndarray | xr.DataArray,
    frequency: np.ndarray | xr.DataArray,
    k_range: tuple[float, float],
    period_range: tuple[float, float],
    direction: str,
) -> xr.DataArray:
    """Binary mask over (frequency, wavenumber) for a simple k-T box.

    k_range     (k_min, k_max)     magnitudes of integer zonal wavenumber.
    period_range (T_min, T_max)    days; frequency bounds are 1/T_max and 1/T_min.
    direction   'eastward' → sign(k)*sign(f) < 0
                'westward' → sign(k)*sign(f) > 0
                'both'     → symmetric (useful as a sanity switch)

    The f=0 row and k=0 column are excluded (no stationary modes).
    """
    k = np.asarray(wavenumber)
    f = np.asarray(frequency)
    K, F = np.meshgrid(k, f)  # K,F have shape (nfreq, nwave)

    k_min, k_max = k_range
    T_min, T_max = period_range
    f_lo, f_hi = 1.0 / T_max, 1.0 / T_min

    abs_k = np.abs(K)
    abs_f = np.abs(F)
    in_k = (abs_k >= k_min) & (abs_k <= k_max)
    in_f = (abs_f >= f_lo) & (abs_f <= f_hi)
    nonzero = (K != 0) & (F != 0)

    if direction == "eastward":
        dir_mask = (K * F) < 0
    elif direction == "westward":
        dir_mask = (K * F) > 0
    elif direction == "both":
        dir_mask = np.ones_like(K, dtype=bool)
    else:
        raise ValueError(f"direction must be eastward/westward/both; got {direction!r}")

    mask = (in_k & in_f & nonzero & dir_mask).astype(np.float64)
    return xr.DataArray(
        mask,
        dims=("frequency", "wavenumber"),
        coords={"frequency": f, "wavenumber": k},
        attrs=dict(k_range=k_range, period_range=period_range, direction=direction),
    )


def apply_mask_ifft(
    fft_data: xr.DataArray,
    mask: xr.DataArray,
    imag_tol: float = 1e-8,
) -> xr.DataArray:
    """Multiply by mask, inverse-FFT to (time, lon), return the real part.

    Raises if the recovered imaginary part is larger than ``imag_tol`` relative to
    the real-part range — that signals a non-Hermitian mask and a buggy filter.
    """
    t_ax = _axis_index(fft_data, "frequency")
    x_ax = _axis_index(fft_data, "wavenumber")

    # Broadcast mask onto fft_data (adds any extra dims like 'lat' automatically).
    masked = fft_data * mask
    y = np.fft.ifft2(masked.values, axes=(t_ax, x_ax))

    real_range = np.ptp(y.real)
    imag_amp = float(np.max(np.abs(y.imag)))
    if real_range > 0 and imag_amp > imag_tol * real_range:
        raise AssertionError(
            f"inverse FFT imaginary residual too large: {imag_amp:.3e} "
            f"(real range {real_range:.3e}) — mask likely not Hermitian symmetric"
        )

    new_dims = tuple(("time" if d == "frequency" else "lon" if d == "wavenumber" else d)
                     for d in fft_data.dims)
    # Caller restores the time coord via trim_padding; here we use integer indices.
    coords = {}
    for d, nd in zip(fft_data.dims, new_dims):
        if d == "frequency":
            coords["time"] = np.arange(fft_data.sizes["frequency"])
        elif d == "wavenumber":
            coords["lon"] = np.arange(fft_data.sizes["wavenumber"])
        else:
            coords[d] = fft_data.coords[d]
    return xr.DataArray(y.real, dims=new_dims, coords=coords)


def trim_padding(
    filtered: xr.DataArray,
    original_len: int,
    reference_time: xr.DataArray,
    reference_lon: xr.DataArray,
) -> xr.DataArray:
    """Drop trailing padding and restore the original time/lon coordinates."""
    out = filtered.isel(time=slice(0, original_len))
    out = out.assign_coords(time=reference_time.values, lon=reference_lon.values)
    return out


def filter_wave(
    data: xr.DataArray,
    wave_type: str,
    spd: float = 4.0,
    pad_zeros: int = 700,
    detrend: bool = False,
    remove_mean: bool = True,
    lat_bounds: tuple[float, float] | None = None,
    filter_spec: Mapping | None = None,
) -> xr.DataArray:
    """Filter a (time, lon) or (time, lat, lon) field to a single wave band.

    ``wave_type`` selects an entry from ``RIOS_BERRIOS_FILTERS`` unless
    ``filter_spec`` is given explicitly (dict with k_range, period_range, direction).
    If the input is 3D with a 'lat' dim, the field is meridionally averaged over
    ``lat_bounds`` (inclusive) before filtering; use ``lat_bounds=None`` to average
    over all available latitudes.
    """
    spec = filter_spec or RIOS_BERRIOS_FILTERS[wave_type]

    if "lat" in data.dims:
        if lat_bounds is not None:
            lo, hi = lat_bounds
            data = data.sel(lat=slice(lo, hi))
        data = data.mean(dim="lat")

    padded, n_time_orig = prepare_for_filter(
        data, pad_zeros=pad_zeros, detrend=detrend, remove_mean=remove_mean,
    )
    z = fft_spacetime(padded, spd=spd)
    mask = build_box_mask(z.wavenumber, z.frequency, **spec)
    filtered_padded = apply_mask_ifft(z, mask)
    out = trim_padding(
        filtered_padded,
        original_len=n_time_orig,
        reference_time=data.time,
        reference_lon=data.lon,
    )
    out.name = f"{data.name or 'field'}_{wave_type}"
    out.attrs.update(
        wave_type=wave_type,
        k_range=tuple(spec["k_range"]),
        period_range=tuple(spec["period_range"]),
        direction=spec["direction"],
        pad_zeros=pad_zeros,
        spd=spd,
    )
    return out


def wave_variance(
    data: xr.DataArray,
    wave_types: Sequence[str] = tuple(RIOS_BERRIOS_FILTERS.keys()),
    spd: float = 4.0,
    pad_zeros: int = 700,
    lat_bounds: tuple[float, float] | None = (-10, 10),
    detrend: bool = False,
) -> xr.Dataset:
    """Variance of raw and wave-filtered fields after meridional average.

    Returns a Dataset with:
      total_variance    scalar, var of the (mean-removed) meridionally-averaged field
      filtered_variance (wave_type,), var of each filtered field
    """
    if "lat" in data.dims:
        if lat_bounds is not None:
            data = data.sel(lat=slice(*lat_bounds))
        merid = data.mean(dim="lat")
    else:
        merid = data

    total_var = float(((merid - merid.mean(dim="time")) ** 2).mean().values)

    var_by_type = []
    for wt in wave_types:
        f = filter_wave(
            merid, wt, spd=spd, pad_zeros=pad_zeros, detrend=detrend, remove_mean=True,
        )
        var_by_type.append(float((f ** 2).mean().values))  # filtered field has zero mean

    out = xr.Dataset(
        data_vars=dict(
            total_variance=xr.DataArray(total_var),
            filtered_variance=xr.DataArray(
                np.asarray(var_by_type), dims=("wave_type",),
                coords={"wave_type": list(wave_types)},
            ),
        ),
        attrs=dict(
            spd=spd, pad_zeros=pad_zeros,
            lat_bounds=str(lat_bounds), detrend=int(bool(detrend)),
        ),
    )
    return out


# ─── Future-proof: Wheeler-Kiladis dispersion-curve masks ──────────────────────

def build_dispersion_mask(
    wavenumber: np.ndarray | xr.DataArray,
    frequency: np.ndarray | xr.DataArray,
    wave_type: str,
) -> xr.DataArray:
    """Wrapper over existing dispersion-curve masks in wavenumber_frequency_functions.

    Not used for the Rios-Berrios 2022 reproduction; provided so callers can swap
    in a Wheeler-Kiladis filter (bounded by equivalent-depth dispersion curves) if
    they want tighter regions than the simple k-T box.
    """
    from wavenumber_frequency_functions import (
        kelvin_wave_mask,
        equatorial_rossby_wave_mask,
        mrg_wave_mask,
    )

    wn_da = wavenumber if isinstance(wavenumber, xr.DataArray) else xr.DataArray(
        wavenumber, dims=("wavenumber",), coords={"wavenumber": wavenumber},
    )
    freq_da = frequency if isinstance(frequency, xr.DataArray) else xr.DataArray(
        frequency, dims=("frequency",), coords={"frequency": frequency},
    )
    builders = {
        "kelvin": kelvin_wave_mask,
        "er": equatorial_rossby_wave_mask,
        "mrg": mrg_wave_mask,
    }
    try:
        builder = builders[wave_type]
    except KeyError as e:
        raise ValueError(
            f"dispersion mask only supported for {list(builders)}; got {wave_type!r}"
        ) from e
    m = builder(wn_da, freq_da).astype(np.float64)
    return m.transpose("frequency", "wavenumber")
