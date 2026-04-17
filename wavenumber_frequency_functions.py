import xarray as xr
import numpy as np
from scipy.signal import convolve2d, detrend
import time
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.DEBUG)

# module data
radius = 6.37122e06  # [m]   average radius of earth
g = 9.80665  # [m/s] gravity at 45 deg lat used by the WMO
omega = 7.292e-05  # [1/s] earth's angular vel


def helper():
    """Prints all the functions that are included in this module."""
    f = [
        "decompose2SymAsym(arr)",
        "rmvAnnualCycle(data, spd, fCrit)",
        "convolvePosNeg(arr, k, dim, boundary_index)",
        "simple_smooth_kernel()",
        "smooth_wavefreq(data, kern=None, nsmooth=None, freq_ax=None, freq_name=None)",
        "resolveWavesHayashi( varfft: xr.DataArray, nDayWin: int, spd: int ) -> xr.DataArray",
        "split_hann_taper(series_length, fraction)",
        "spacetime_power(data, segsize=96, noverlap=60, spd=1, latitude_bounds=None, dosymmetries=False, rmvLowFrq=False)",
        "genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12])",
    ]
    [print(fe) for fe in f]


@contextmanager
def optional_timer(enabled=True, name=""):
    """Optional timer to include for profiling bits of code."""
    if enabled:
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.6f} seconds")
    else:
        yield


def decompose2SymAsym(arr):
    """Mimic NCL function to decompose into symmetric and asymmetric parts.

    arr: xarra DataArray

    return: DataArray with symmetric in SH, asymmetric in NH

    Note:
        This function produces indistinguishable results from NCL version.
    """
    lat_dim = arr.dims.index("lat")
    # flag to follow NCL convention and put symmetric component in SH
    # & asymmetric in NH
    # method: use flip to reverse latitude, put in DataArray for coords, use loc/isel
    # to assign to negative/positive latitudes (exact equator is left alone)
    data_sym = 0.5 * (arr.values + np.flip(arr.values, axis=lat_dim))
    data_asy = 0.5 * (arr.values - np.flip(arr.values, axis=lat_dim))
    data_sym = xr.DataArray(data_sym, dims=arr.dims, coords=arr.coords)
    data_asy = xr.DataArray(data_asy, dims=arr.dims, coords=arr.coords)
    out = arr.copy()  # might not be best to copy, but is safe
    out.loc[{"lat": arr["lat"][arr["lat"] < 0]}] = data_sym.isel(lat=data_sym.lat < 0)
    out.loc[{"lat": arr["lat"][arr["lat"] > 0]}] = data_asy.isel(lat=data_asy.lat > 0)
    return out


def rmvAnnualCycle(data, spd, fCrit):
    """remove frequencies less than fCrit from data.

    data: xarray DataArray
    spd: sampling frequency in samples-per-day
    fCrit: frequency threshold; remove frequencies < fCrit

    return: xarray DataArray, shape of data

    Note: fft/ifft preserves the mean because z = fft(x), z[0] is the mean.
          To keep the mean here, we need to keep the 0 frequency.

    Note: This function reproduces the results from the NCL version.

    Note: Two methods are available, one using fft/ifft and the other rfft/irfft.
          They both produce output that is indistinguishable from NCL's result.
    """
    dimz = data.sizes
    ntim = dimz["time"]
    time_ax = list(data.dims).index("time")
    # Method 1: Uses the complex FFT, returns the negative frequencies too, but they
    # should be redundant b/c they are conjugate of positive ones.
    cf = np.fft.fft(data.values, axis=time_ax)
    freq = np.fft.fftfreq(ntim, spd)
    cf[(freq != 0) & (np.abs(freq) < fCrit), ...] = 0.0  # keeps the mean
    z = np.fft.ifft(cf, n=ntim, axis=0)
    # Method 2: Uses the real FFT. In this case,
    # cf = np.fft.rfft(data.values, axis=time_ax)
    # freq = np.linspace(1, (ntim*spd)//2, (ntim*spd)//2) / ntim
    # fcrit_ndx = np.argwhere(freq < fCrit).max()
    # if fcrit_ndx > 1:
    #     cf[1:fcrit_ndx+1, ...] = 0.0
    # z = np.fft.irfft(cf, n=ntim, axis=0)
    z = xr.DataArray(z.real, dims=data.dims, coords=data.coords)
    return z


def convolvePosNeg(arr, k, dim, boundary_index):
    """Apply convolution of (arr, k) excluding data at boundary_index in dimension dim.

    arr: numpy ndarray of data
    k: numpy ndarray, same dimension as arr, this should be the kernel
    dim: integer indicating the axis of arr to split
    boundary_index: integer indicating the position to split dim

    Split array along dim at boundary_index;
    perform convolution on each sub-array;
    reconstruct output array from the two subarrays;
    the values of output at boundary_index of dim will be same as input.

    `convolve2d` is `scipy.signal.convolve2d()`
    """
    # arr: numpy ndarray
    oarr = arr.copy()  # maybe not good to make a fresh copy every time?
    # first pass is [0 : boundary_index)
    slc1 = [slice(None)] * arr.ndim
    slc1[dim] = slice(None, boundary_index)
    arr1 = arr[tuple(slc1)]
    ans1 = convolve2d(arr1, k, boundary="symm", mode="same")
    # second pass is [boundary_index+1, end]
    slc2 = [slice(None)] * arr.ndim
    slc2[dim] = slice(boundary_index + 1, None)
    arr2 = arr[tuple(slc2)]
    ans2 = convolve2d(arr2, k, boundary="symm", mode="same")
    # fill in the output array
    oarr[tuple(slc1)] = ans1
    oarr[tuple(slc2)] = ans2
    return oarr


def simple_smooth_kernel():
    """Provide a very simple smoothing kernel."""
    kern = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
    return kern / kern.sum()


def smooth_wavefreq(data, kern=None, nsmooth=None, freq_ax=None, freq_name=None):
    """Apply a convolution of (data,kern) nsmooth times.
    The convolution is applied separately to the positive and negative frequencies.
    Either the name (freq_name: str) or axis index (freq_ax: int) of frequency is required, with the name preferred.
    """
    assert isinstance(data, xr.DataArray)
    if kern is None:
        kern = simple_smooth_kernel()
    if nsmooth is None:
        nsmooth = 20
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        nzero = (
            data.sizes[freq_name] // 2
        )  # <-- THIS IS SUPPOSED TO BE THE INDEX AT FREQ==0.0
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
    else:
        raise ValueError(
            "smooth_wavefreq needs to know how to find frequency dimension."
        )
    smth1pass = convolvePosNeg(
        data, kern, axnum, nzero
    )  # this is a custom function to skip 0-frequency (mean)
    # note: the convolution is strictly 2D and the boundary condition is symmetric --> if kernel is normalized, preserves the sum.
    smth1pass = xr.DataArray(
        smth1pass, dims=data.dims, coords=data.coords
    )  # ~copy_metadata
    # repeat smoothing many times:
    smthNpass = smth1pass.values.copy()
    for i in range(nsmooth):
        smthNpass = convolvePosNeg(smthNpass, kern, axnum, nzero)
    return xr.DataArray(smthNpass, dims=data.dims, coords=data.coords)


def smooth_wavefreq_frequency_dependent(data, kern=None, freq_bands=None, nsmooth_bands=None, freq_ax=None, freq_name=None):
    """Apply frequency-dependent smoothing to wavenumber-frequency spectrum.

    This function applies different numbers of smoothing passes to different frequency bands,
    following the approach from tropical_diagnostics where more smoothing is applied at low
    frequencies to better capture the background spectral envelope.

    Parameters
    ----------
    data : xarray.DataArray
        Input wavenumber-frequency spectrum to smooth
    kern : numpy.ndarray, optional
        Smoothing kernel. If None, uses simple_smooth_kernel() (default: None)
    freq_bands : list of float, optional
        Frequency band boundaries in cpd. Default: [0.0, 0.2, 0.5, 2.0]
        Creates bands: [0.0-0.2), [0.2-0.5), [0.5-2.0]
    nsmooth_bands : list of int, optional
        Number of smoothing passes for each frequency band.
        Default: [100, 50, 20] (more smoothing at low frequencies)
        Length must be len(freq_bands) - 1
    freq_ax : int, optional
        Axis index of frequency dimension (default: None)
    freq_name : str, optional
        Name of frequency dimension, preferred over freq_ax (default: None)

    Returns
    -------
    xarray.DataArray
        Smoothed spectrum with same dimensions and coordinates as input

    Notes
    -----
    - Applies convolution separately to positive and negative frequencies
    - Zero frequency is preserved (not smoothed)
    - Default settings inspired by tropical_diagnostics approach:
      * Low frequencies (< 0.2 cpd): 100 smoothing passes
      * Mid frequencies (0.2-0.5 cpd): 50 passes
      * High frequencies (> 0.5 cpd): 20 passes
    - This helps the background better capture the spectral envelope at low frequencies
      where sharp drop-offs can cause artifacts with uniform smoothing

    Examples
    --------
    >>> # Use default frequency-dependent smoothing
    >>> bg = smooth_wavefreq_frequency_dependent(spectrum, freq_name='frequency')
    >>>
    >>> # Custom frequency bands and smoothing
    >>> bg = smooth_wavefreq_frequency_dependent(
    ...     spectrum,
    ...     freq_bands=[0.0, 0.1, 0.3, 1.0],
    ...     nsmooth_bands=[150, 75, 30],
    ...     freq_name='frequency'
    ... )
    """
    assert isinstance(data, xr.DataArray), "Input data must be xarray.DataArray"

    # Set defaults
    if kern is None:
        kern = simple_smooth_kernel()

    if freq_bands is None:
        # Default bands based on tropical_diagnostics approach
        freq_bands = [0.0, 0.2, 0.5, 2.0]

    if nsmooth_bands is None:
        # Default: more smoothing at low frequencies
        nsmooth_bands = [100, 50, 20]

    # Validate inputs
    if len(nsmooth_bands) != len(freq_bands) - 1:
        raise ValueError(
            f"nsmooth_bands length ({len(nsmooth_bands)}) must be "
            f"len(freq_bands) - 1 ({len(freq_bands) - 1})"
        )

    # Get frequency dimension info
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        nzero = data.sizes[freq_name] // 2  # Index at freq == 0.0
        freq_coord = data[freq_name].values
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
        freq_coord = data.coords[data.dims[freq_ax]].values
    else:
        raise ValueError(
            "smooth_wavefreq_frequency_dependent needs freq_name or freq_ax"
        )

    # Sort bands by number of smoothing passes (ascending)
    # This allows us to apply smoothing incrementally
    sorted_indices = np.argsort(nsmooth_bands)

    # Create mapping of frequency index to target smoothing passes
    freq_abs = np.abs(freq_coord)
    target_smoothing = np.zeros(len(freq_coord), dtype=int)

    for i, (f_low, f_high) in enumerate(zip(freq_bands[:-1], freq_bands[1:])):
        band_mask = (freq_abs >= f_low) & (freq_abs < f_high)
        target_smoothing[band_mask] = nsmooth_bands[i]

    # Initialize with input data
    result = data.values.copy()

    # Apply smoothing incrementally
    # Start with minimum smoothing, then add more for bands that need it
    smoothing_levels = sorted(set(nsmooth_bands))

    for level_idx, current_level in enumerate(smoothing_levels):
        if level_idx == 0:
            # Apply initial smoothing passes
            n_passes = current_level
        else:
            # Apply additional passes beyond previous level
            n_passes = current_level - smoothing_levels[level_idx - 1]

        # Apply n_passes of smoothing to entire array
        for _ in range(n_passes):
            result = convolvePosNeg(result, kern, axnum, nzero)

        # If this is not the maximum level, we need to save this state
        # for frequencies that should have exactly this much smoothing
        if level_idx < len(smoothing_levels) - 1:
            # Identify which frequencies should stop at this level
            freq_mask = (target_smoothing == current_level)

            if np.any(freq_mask):
                # Save current smoothed state for these frequencies
                # We'll need to restore them after further smoothing
                slc = [slice(None)] * result.ndim
                slc[axnum] = freq_mask
                # Store this in a temporary array for later restoration
                if level_idx == 0:
                    saved_states = {}
                saved_states[current_level] = result[tuple(slc)].copy()

    # Now restore the saved states for frequencies that needed less smoothing
    if len(smoothing_levels) > 1:
        for level in smoothing_levels[:-1]:  # Skip the maximum level
            freq_mask = (target_smoothing == level)
            if np.any(freq_mask):
                slc = [slice(None)] * result.ndim
                slc[axnum] = freq_mask
                result[tuple(slc)] = saved_states[level]

    return xr.DataArray(result, dims=data.dims, coords=data.coords)


def estimate_background_parametric(data, method='polynomial', poly_degree=3,
                                   initial_smooth_passes=20, freq_ax=None, freq_name=None):
    """Estimate background spectrum using parametric model fitting.

    This function fits a smooth parametric model to the spectrum to estimate the
    background, which can better capture gradual spectral envelopes than pure smoothing,
    especially when the spectrum has sharp features at low frequencies.

    Parameters
    ----------
    data : xarray.DataArray
        Input wavenumber-frequency spectrum
    method : str, optional
        Parametric model type. Options:
        - 'polynomial': 2D polynomial surface fit (default)
        - 'rednoise': Red-noise model fit
        - 'hybrid': Polynomial at low freq, smoothing at high freq
    poly_degree : int, optional
        Degree of polynomial for 'polynomial' method (default: 3)
        Higher degree captures more spectral structure
    initial_smooth_passes : int, optional
        Number of smoothing passes to apply before fitting (default: 20)
        Removes fine-scale wave features before model fitting
    freq_ax : int, optional
        Axis index of frequency dimension
    freq_name : str, optional
        Name of frequency dimension, preferred over freq_ax

    Returns
    -------
    xarray.DataArray
        Estimated background spectrum with same dimensions as input

    Notes
    -----
    - First applies moderate smoothing to remove wave signatures
    - Then fits parametric model to capture broad spectral envelope
    - Handles sharp spectral features better than pure smoothing
    - Zero frequency is excluded from fitting and set to original value

    The polynomial method fits:
        log(P) = sum_{i,j} c_ij * k^i * f^j
    where k is wavenumber, f is frequency, and c_ij are coefficients

    Examples
    --------
    >>> # Polynomial background (recommended for sharp low-freq features)
    >>> bg = estimate_background_parametric(spectrum, method='polynomial',
    ...                                      poly_degree=3, freq_name='frequency')
    >>>
    >>> # Hybrid approach
    >>> bg = estimate_background_parametric(spectrum, method='hybrid',
    ...                                      freq_name='frequency')
    """
    assert isinstance(data, xr.DataArray), "Input must be xarray.DataArray"

    # Get frequency dimension info
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        nzero = data.sizes[freq_name] // 2
        freq_coord = data[freq_name].values
        freq_dim_name = freq_name
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
        freq_dim_name = data.dims[freq_ax]
        freq_coord = data.coords[freq_dim_name].values
    else:
        raise ValueError("Need freq_name or freq_ax")

    # Get wavenumber dimension (assume it's the other dimension for 2D data)
    wn_dim_idx = 1 - axnum  # For 2D: if freq is dim 0, wn is dim 1, and vice versa
    wn_dim_name = data.dims[wn_dim_idx]
    wn_coord = data[wn_dim_name].values

    # Step 1: Apply initial smoothing to remove wave features
    print(f"  Applying {initial_smooth_passes} smoothing passes...")
    smoothed = smooth_wavefreq(data, nsmooth=initial_smooth_passes,
                               freq_name=freq_dim_name)

    if method == 'polynomial':
        print(f"  Fitting {poly_degree}-degree 2D polynomial model...")
        background = _fit_polynomial_background(
            smoothed, wn_coord, freq_coord, poly_degree,
            axnum, nzero, freq_dim_name, wn_dim_name
        )
    elif method == 'rednoise':
        print("  Fitting red-noise model...")
        background = _fit_rednoise_background(
            smoothed, wn_coord, freq_coord,
            axnum, nzero, freq_dim_name, wn_dim_name
        )
    elif method == 'hybrid':
        print("  Fitting hybrid model (polynomial low-freq, smoothing high-freq)...")
        background = _fit_hybrid_background(
            smoothed, data, wn_coord, freq_coord, poly_degree,
            axnum, nzero, freq_dim_name, wn_dim_name
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return background


def _fit_polynomial_background(data, wn_coord, freq_coord, degree, freq_ax, nzero, freq_dim, wn_dim):
    """Fit 2D polynomial to log-spectrum."""
    from scipy import optimize

    # Create meshgrid
    if freq_ax == 0:
        # data is (frequency, wavenumber)
        freq_mesh, wn_mesh = np.meshgrid(freq_coord, wn_coord, indexing='ij')
    else:
        # data is (wavenumber, frequency)
        wn_mesh, freq_mesh = np.meshgrid(wn_coord, freq_coord, indexing='ij')

    # Use absolute frequency for fitting (symmetric about zero)
    freq_abs = np.abs(freq_mesh)

    # Exclude zero frequency and any NaN/inf values
    valid_mask = (freq_abs > 0) & np.isfinite(data.values) & (data.values > 0)

    # Take log of spectrum for fitting (red noise is linear in log space)
    log_data = np.log(data.values)

    # Prepare data for fitting
    freq_fit = freq_abs[valid_mask].flatten()
    wn_fit = wn_mesh[valid_mask].flatten()
    log_power_fit = log_data[valid_mask].flatten()

    # Normalize coordinates for better numerical stability
    freq_scale = np.max(freq_abs)
    wn_scale = np.max(np.abs(wn_mesh))
    freq_norm = freq_fit / freq_scale
    wn_norm = wn_fit / wn_scale

    # Build design matrix for polynomial fit
    # Fit: log(P) = c00 + c10*k + c01*f + c20*k^2 + c11*k*f + c02*f^2 + ...
    n_terms = (degree + 1) * (degree + 2) // 2  # Number of terms in 2D polynomial
    X = np.zeros((len(freq_norm), n_terms))

    term_idx = 0
    for total_degree in range(degree + 1):
        for k_deg in range(total_degree + 1):
            f_deg = total_degree - k_deg
            X[:, term_idx] = (wn_norm ** k_deg) * (freq_norm ** f_deg)
            term_idx += 1

    # Solve least squares: X * coeffs = log_power
    coeffs, residuals, rank, s = np.linalg.lstsq(X, log_power_fit, rcond=None)

    # Evaluate polynomial on full grid
    X_full = np.zeros((freq_abs.size, n_terms))
    freq_norm_full = (freq_abs / freq_scale).flatten()
    wn_norm_full = (wn_mesh / wn_scale).flatten()

    term_idx = 0
    for total_degree in range(degree + 1):
        for k_deg in range(total_degree + 1):
            f_deg = total_degree - k_deg
            X_full[:, term_idx] = (wn_norm_full ** k_deg) * (freq_norm_full ** f_deg)
            term_idx += 1

    log_background = (X_full @ coeffs).reshape(data.shape)
    background = np.exp(log_background)

    # Set zero frequency to NaN (will be handled by normalization later)
    if freq_ax == 0:
        background[nzero, :] = np.nan
    else:
        background[:, nzero] = np.nan

    return xr.DataArray(background, dims=data.dims, coords=data.coords)


def _fit_rednoise_background(data, wn_coord, freq_coord, freq_ax, nzero, freq_dim, wn_dim):
    """Fit red-noise model: P(k,f) = A(k) / (1 + (f/f0(k))^beta)."""
    # This is a simplified version - fit frequency spectrum for each wavenumber

    if freq_ax == 0:
        # data is (frequency, wavenumber)
        background = np.zeros_like(data.values)

        for k_idx in range(len(wn_coord)):
            freq_spectrum = data.values[:, k_idx]
            valid_mask = (np.abs(freq_coord) > 0) & np.isfinite(freq_spectrum) & (freq_spectrum > 0)

            if np.sum(valid_mask) > 3:
                # Fit simple power law: P ~ f^(-beta)
                freq_valid = np.abs(freq_coord[valid_mask])
                power_valid = freq_spectrum[valid_mask]

                # Log-linear fit
                log_f = np.log(freq_valid)
                log_p = np.log(power_valid)
                coeffs = np.polyfit(log_f, log_p, 1)  # Linear in log space

                # Evaluate on full frequency range (excluding zero)
                log_bg = coeffs[0] * np.log(np.abs(freq_coord) + 1e-10) + coeffs[1]
                background[:, k_idx] = np.exp(log_bg)
            else:
                background[:, k_idx] = np.nanmean(freq_spectrum)

            background[nzero, k_idx] = np.nan
    else:
        # data is (wavenumber, frequency)
        background = np.zeros_like(data.values)

        for k_idx in range(len(wn_coord)):
            freq_spectrum = data.values[k_idx, :]
            valid_mask = (np.abs(freq_coord) > 0) & np.isfinite(freq_spectrum) & (freq_spectrum > 0)

            if np.sum(valid_mask) > 3:
                freq_valid = np.abs(freq_coord[valid_mask])
                power_valid = freq_spectrum[valid_mask]

                log_f = np.log(freq_valid)
                log_p = np.log(power_valid)
                coeffs = np.polyfit(log_f, log_p, 1)

                log_bg = coeffs[0] * np.log(np.abs(freq_coord) + 1e-10) + coeffs[1]
                background[k_idx, :] = np.exp(log_bg)
            else:
                background[k_idx, :] = np.nanmean(freq_spectrum)

            background[k_idx, nzero] = np.nan

    return xr.DataArray(background, dims=data.dims, coords=data.coords)


def _fit_hybrid_background(smoothed, original, wn_coord, freq_coord, poly_degree,
                           freq_ax, nzero, freq_dim, wn_dim):
    """Hybrid: extrapolate from smoothed background at very low freq, use smoothing elsewhere."""
    # Only fit to very low frequencies to avoid polynomial extrapolation issues
    # Transition frequency (cpd) - use polynomial only below this
    f_fit_max = 0.05  # Only fit polynomial to frequencies below 0.05 cpd
    f_transition = 0.08  # Blend to pure smoothing by 0.08 cpd

    # Fit polynomial ONLY to low-frequency portion of the smoothed spectrum
    freq_abs = np.abs(freq_coord)

    # Create a modified smoothed spectrum where we only fit low frequencies
    low_freq_mask = freq_abs <= f_fit_max

    # For polynomial fitting, we'll fit only to the low-frequency region
    # but evaluate on all frequencies
    if freq_ax == 0:
        # data is (frequency, wavenumber)
        freq_mesh, wn_mesh = np.meshgrid(freq_coord, wn_coord, indexing='ij')
    else:
        # data is (wavenumber, frequency)
        wn_mesh, freq_mesh = np.meshgrid(wn_coord, freq_coord, indexing='ij')

    freq_abs_mesh = np.abs(freq_mesh)

    # Build constraint: polynomial should match smoothed background at f_fit_max
    # Simple approach: fit polynomial to smoothed values at low freq
    valid_mask = (freq_abs_mesh > 0) & (freq_abs_mesh <= f_fit_max) & np.isfinite(smoothed.values) & (smoothed.values > 0)

    if np.sum(valid_mask) < 10:
        # Not enough points, just return smoothed
        return smoothed

    log_data = np.log(smoothed.values)

    freq_fit = freq_abs_mesh[valid_mask].flatten()
    wn_fit = wn_mesh[valid_mask].flatten()
    log_power_fit = log_data[valid_mask].flatten()

    # Normalize
    freq_scale = f_fit_max
    wn_scale = np.max(np.abs(wn_mesh))
    freq_norm = freq_fit / freq_scale
    wn_norm = wn_fit / wn_scale

    # Use simpler polynomial (degree 2 or less) to avoid overfitting
    fit_degree = min(poly_degree, 2)
    n_terms = (fit_degree + 1) * (fit_degree + 2) // 2
    X = np.zeros((len(freq_norm), n_terms))

    term_idx = 0
    for total_degree in range(fit_degree + 1):
        for k_deg in range(total_degree + 1):
            f_deg = total_degree - k_deg
            X[:, term_idx] = (wn_norm ** k_deg) * (freq_norm ** f_deg)
            term_idx += 1

    coeffs, residuals, rank, s = np.linalg.lstsq(X, log_power_fit, rcond=None)

    # Evaluate polynomial on full grid
    X_full = np.zeros((freq_abs_mesh.size, n_terms))
    freq_norm_full = (freq_abs_mesh / freq_scale).flatten()
    wn_norm_full = (wn_mesh / wn_scale).flatten()

    term_idx = 0
    for total_degree in range(fit_degree + 1):
        for k_deg in range(total_degree + 1):
            f_deg = total_degree - k_deg
            X_full[:, term_idx] = (wn_norm_full ** k_deg) * (freq_norm_full ** f_deg)
            term_idx += 1

    log_poly_bg = (X_full @ coeffs).reshape(smoothed.shape)
    poly_bg = np.exp(log_poly_bg)

    # Blend polynomial and smoothed backgrounds
    # Use polynomial at very low freq, transition to smoothed
    weights = (freq_abs_mesh - f_fit_max) / (f_transition - f_fit_max)
    weights = np.clip(weights, 0, 1)

    # Smooth the weights to avoid sharp transitions
    from scipy.ndimage import gaussian_filter1d
    if freq_ax == 0:
        weights = gaussian_filter1d(weights, sigma=2, axis=0)
    else:
        weights = gaussian_filter1d(weights, sigma=2, axis=1)

    background = (1 - weights) * poly_bg + weights * smoothed.values

    # Set zero frequency to NaN
    if freq_ax == 0:
        background[nzero, :] = np.nan
    else:
        background[:, nzero] = np.nan

    return xr.DataArray(background, dims=smoothed.dims, coords=smoothed.coords)


# def smooth_gaussian(data, sigma=1, freq_ax=None, freq_name=None):
#     '''smoothing using a gaussian filter (from Isla)'''
#     # fill in k=0 with the average of adjacent values for the purposes of smoothing
#     datfill = (data.sel(k=-1).values + data.sel(k=1))/2.
#     dat = data.where( dat.k != 0, datfill )
#     # drop w=0
#     dat = dat.where( dat.w !=0, drop=True)
#     dat_smooth = gaussian_filter(dat, sigma=sigma)
#     dat_smooth = xr.DataArray(dat_smooth, coords=dat.coords, dims=dat.dims)
#     return dat_smooth


def resolveWavesHayashi(varfft: xr.DataArray, nDayWin: int, spd: int) -> xr.DataArray:
    """This is a direct translation from the NCL routine to python/xarray.
    input:
        varfft : expected to have rightmost dimensions of wavenumber and frequency.
        varfft : expected to be an xarray DataArray with coordinate variables.
        nDayWin : integer that is the length of the segments in days.
        spd : the sampling frequency in `timesteps` per day (I think).

    returns:
        a DataArray that is reordered to have correct westward & eastward propagation.

    """
    # -------------------------------------------------------------
    # Special reordering to resolve the Progressive and Retrogressive waves
    # Reference: Hayashi, Y.
    #    A Generalized Method of Resolving Disturbances into
    #    Progressive and Retrogressive Waves by Space and
    #    Fourier and Time Cross-Spectral Analysis
    #    J. Meteor. Soc. Japan, 1971, 49: 125-128.
    # -------------------------------------------------------------

    # in NCL varfft is dimensioned (2,mlon,nSampWin), but the first dim doesn't matter b/c python supports complex numbers.
    #
    # Create array PEE(NL+1,NT+1) which contains the (real) power spectrum.
    # all the following assume indexing starting with 0
    # In this array (PEE), the negative wavenumbers will be from pn=0 to NL/2-1 (left).
    # The positive wavenumbers will be for pn=NL/2+1 to NL (right).
    # Negative frequencies will be from pt=0 to NT/2-1 (left).
    # Positive frequencies will be from pt=NT/2+1 to NT  (right).
    # Information about zonal mean will be for pn=NL/2 (middle).
    # Information about time mean will be for pt=NT/2 (middle).
    # Information about the Nyquist Frequency is at pt=0 and pt=NT
    #

    # In PEE, define the
    # WESTWARD waves to be either
    #          positive frequency and negative wavenumber
    #          OR
    #          negative freq and positive wavenumber.
    # EASTWARD waves are either positive freq and positive wavenumber
    #          OR negative freq and negative wavenumber.

    # Note that frequencies are returned from fftpack are ordered like so
    #    input_time_pos [ 0    1   2    3     4      5    6   7  ]
    #    ouput_fft_coef [mean 1/7 2/7  3/7 nyquist -3/7 -2/7 -1/7]
    #                    mean,pos freq to nyq,neg freq hi to lo
    #
    # Rearrange the coef array to give you power array of freq and wave number east/west
    # Note east/west wave number *NOT* eq to fft wavenumber see Hayashi '71
    # Hence, NCL's 'cfftf_frq_reorder' can *not* be used.
    # BPM: This goes for np.fft.fftshift
    #
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array varfft(2,n,t)   dimensioned (2,0:numlon-1,0:numtim-1)
    # new space/time pee(2,pn,pt) dimensioned (2,0:numlon  ,0:numtim  )
    #
    # NOTE: one larger in both freq/space dims
    # the initial index of 2 is for the real (indx 0) and imag (indx 1) parts of the array
    #
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2

    # local variables : dimvf, numlon, N, varspacetime, pee, wave, freq

    logging.debug(f"[Hayashi] nDayWin: {nDayWin}, spd: {spd}")
    dimnames = varfft.dims
    dimvf = varfft.shape
    mlon = len(varfft["wavenumber"])  # number of longitudes = numer of wavenumbers
    N = len(varfft["frequency"])
    k_dim_index = dimnames.index("wavenumber")
    f_dim_index = dimnames.index("frequency")
    logging.info(
        f"[Hayashi] input dims is {dimnames}, {dimvf} || Input dtype: {varfft.dtype = }"
    )
    logging.info(f"[Hayashi] input coords is {varfft.coords}")
    logging.debug(
        f"[Hayashi] wavenumber axis is {k_dim_index}, frequency axis is {f_dim_index}"
    )
    if len(dimnames) != len(varfft.coords):
        logging.error("The size of varfft.coords is incorrect.")
        raise ValueError("STOP")

    nshape = list(dimvf)
    nshape[k_dim_index] += 1
    nshape[f_dim_index] += 1
    logging.debug(f"[Hayashi] The nshape ends up being {nshape}")
    # this is a reordering, use Ellipsis to allow arbitrary number of dimensions,
    # but we insist that the wavenumber and frequency dims are rightmost.
    # we will fill the new array in increasing order (arbitrary choice)
    varspacetime = np.full(nshape, np.nan, dtype=type(varfft))
    # first two are the negative wavenumbers (westward),
    # second two are the positive wavenumbers (eastward)
    logging.debug(
        f"[Hayashi] Assign values into array. Notable numbers: mlon//2={mlon//2}, N//2={N//2}"
    )
    varspacetime[..., 0 : mlon // 2, 0 : N // 2] = varfft[
        ..., mlon // 2 : 0 : -1, N // 2 :
    ]  # neg.k, pos.w
    varspacetime[..., 0 : mlon // 2, N // 2 :] = varfft[
        ..., mlon // 2 : 0 : -1, 0 : N // 2 + 1
    ]  # neg.k,
    varspacetime[..., mlon // 2 :, 0 : N // 2 + 1] = varfft[
        ..., 0 : mlon // 2 + 1, N // 2 :: -1
    ]  # assign eastward & neg.freq.
    varspacetime[..., mlon // 2 :, N // 2 + 1 :] = varfft[
        ..., 0 : mlon // 2 + 1, -1 : N // 2 - 1 : -1
    ]  # assign eastward & pos.freq.
    logging.debug(f"[Hayashi] Shape after reordering: {varspacetime.shape}")
    logging.debug(f"[Hayashi] Sum after reordering: {varspacetime.sum()}")
    #  Create the real power spectrum pee = sqrt(real^2+imag^2)^2
    logging.debug(
        f"[Hayashi] calculate power by absolute value (i.e. sqrt(real**2 + imag**2))and squaring."
    )
    pee = (np.abs(varspacetime)) ** 2
    logging.debug(
        f"[Hayashi] sum of pee {pee.sum()}. Type of pee: {type(pee)} Dtype: {pee.dtype}"
    )
    logging.debug(f"[Hayashi] put into DataArray")
    # add meta data for use upon return
    wave = np.arange(-mlon // 2, (mlon // 2) + 1, 1, dtype=int)
    freq = (
        np.linspace(-1 * nDayWin * spd / 2, nDayWin * spd / 2, (nDayWin * spd) + 1)
        / nDayWin
    )

    logging.debug(f"[Hayashi] freq size is {freq.shape}.")
    odims = list(dimnames)
    odims[-2] = "wavenumber"
    odims[-1] = "frequency"
    ocoords = {}
    for c in varfft.coords:
        logging.debug(f"[hayashi] working on coordinate {c}")
        if (c != "wavenumber") and (c != "frequency"):
            ocoords[c] = varfft[c]
        elif c == "wavenumber":
            ocoords["wavenumber"] = wave
        elif c == "frequency":
            ocoords["frequency"] = freq
    pee = xr.DataArray(pee, dims=odims, coords=ocoords)
    z = pee.copy()
    z.loc[{"frequency": 0}] = np.nan
    logging.debug(f"[Hayashi] Sum at the end (removing zero freq): {z.sum().item()}")
    return pee


def split_hann_taper(series_length, fraction):
    """Implements `split cosine bell` taper of length `series_length`
       where only fraction of points are tapered (combined on both ends).

    This returns a function that tapers to zero on the ends. To taper to the mean of a series X:
    XTAPER = (X - X.mean())*series_taper + X.mean()
    """
    npts = int(np.rint(fraction * series_length))  # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0 : npts // 2 + 1] = taper[0 : npts // 2 + 1]
    series_taper[-npts // 2 + 1 :] = taper[npts // 2 + 1 :]
    return series_taper


def spacetime_power(
    data,
    segsize=96,
    noverlap=60,
    spd=1,
    latitude_bounds=None,
    dosymmetries=False,
    rmvLowFrq=False,
    lataggreg='sum'
):
    """Perform space-time spectral decomposition and return power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    segsize: integer denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)

    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.

    dosymmetries: if True, follow NCL convention of putting symmetric component in SH, antisymmetric in NH
                  If True, the function returns a DataArray with a `component` dimension.

    rmvLowFrq: if True, remove frequencies < 1/segsize from data.

    lataggreg: latitude aggregation, function to aggregate spectra at latitudes. NCL uses 'sum', but 'mean' is probably better.

    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data (but keeps the mean value, as in NCL)
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Construct overlapping window view of data.
        6. Detrend the segments (strange enough, removing mean).
        7. Apply taper in time dimension of windows (aka segments).
        8. Fourier transform
        9. Apply Hayashi reordering to get propagation direction & convert to power.
       10. return DataArray with power.

    Notes
    -----
        Upon returning power, this should be comparable to "raw" spectra.
        Next step would be be to smooth with `smooth_wavefreq`,
        and divide raw spectra by smooth background to obtain "significant" spectral power.

    """

    # convert from days to time steps
    segsize = spd*segsize
    noverlap = spd*noverlap

    if latitude_bounds is not None:
        assert isinstance(latitude_bounds, tuple)
        data = data.sel(
            lat=slice(*latitude_bounds)
        )  # CAUTION: is this a mutable argument?
        logging.info(f"Data reduced by latitude bounds. Size is {data.sizes}")
        slat = latitude_bounds[0]
        nlat = latitude_bounds[1]
    else:
        slat = data["lat"].min().item()
        nlat = data["lat"].max().item()

    # "Remove dominant signals"

    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    xmean = data.mean(dim="time")
    xdetr = detrend(data.values, axis=0, type="linear")
    xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    xdetr += xmean  # put the mean back in
    # --> Tested and confirmed that this approach gives same answer as NCL

    # field testing it: pass
    # if not(hasattr(xdetr, "name")) or (not isinstance(xdetr.name, str) ):
    #     xdetr.name = "detrended"
    # xdetr.to_netcdf("/Users/brianpm/Documents/pout_0_detrend.nc")

    # filter low-frequencies
    if rmvLowFrq:
        data = rmvAnnualCycle(xdetr, spd, 1 / segsize)
    # --> Tested and confirmed that this function gives same answer as NCL
    # testing: pass -- indistinguisable from file produced by NCL
    # data.name = "filtered"
    # data.to_netcdf("/Users/brianpm/Documents/pout_1_filtered.nc")

    # NOTE: we have altered "data" to be detrended & filtered at this point

    dimsizes = data.sizes  # dict
    lon_size = dimsizes["lon"]
    lat_size = dimsizes["lat"]
    lat_dim = data.dims.index("lat")
    if dosymmetries:
        data = decompose2SymAsym(data)
    # testing: pass -- Gets the same result as NCL.
    logging.debug(
        f"[spacetime_power] data shape after removing low frequencies: {data.shape}"
    )
    logging.debug(
        f"[spacetime_power] variance of data before windowing: {np.var(data).item()}"
    )

    # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
    # WK99 recommend "2-month" overlap
    # Shape of x_win: (_, lat, lon, segments: spd*segsize)
    x_roll = data.rolling(time=segsize, min_periods=segsize)  # WK99 use 96-day window
    assert (
        segsize - noverlap > 0
    ), f"Error, inconsistent specification of segsize and noverlap results in stride of {segsize-noverlap}, but must be > 0."
    x_win = x_roll.construct("segments")
    x_win = x_win.isel(time=slice(segsize - 1, None, segsize - noverlap))

    logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
    # Additional detrend for each segment:
    if np.logical_not(np.any(np.isnan(x_win))):
        logging.info("No missing, so use simplest segment detrend.")
        x_win_detr = detrend(
            x_win.values, axis=-1, type="linear"
        )  # <-- missing data makes this not work
        x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
    else:
        logging.warning(
            "EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead."
        )
        logging.warning(
            "There are missing data in x_win, so have to try to detrend around them."
        )
        x_win_cp = x_win.values.copy()
        logging.info(
            f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension."
        )
        x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(
            x_win_cp[np.logical_not(np.isnan(x_win_cp))]
        )
        x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
    logging.debug(
        f"[spacetime_power] x_win variance of segments: {np.var(x_win, axis=(1,2,3)).values}"
    )
    # 3. Taper in time to make the signal periodic, as required for FFT.
    # taper = np.hanning(segsize)  # WK seem to use some kind of stretched out hanning window; unclear if it matters
    taper = split_hann_taper(segsize, 0.1)  # try to replicate NCL's
    x_wintap = x_win * taper  # would do XTAPER = (X - X.mean())*series_taper + X.mean()
    # But since we have removed the mean, taper going to 0 is equivalent to taper going to the mean.
    logging.debug(
        f"[spacetime_power] x_wintap variance of segments: {np.var(x_wintap, axis=(1,2,3)).values}"
    )

    # Do the transform using 2D FFT
    # - normalize by dimension sizes
    z = np.fft.fft2(x_wintap, axes=(2, 3)) / (lon_size * segsize)

    # NOTE: with this normalization, the power spectral density should
    #       be calculated as np.abs(z)**2 * dlon * dt * lon_size * segsize
    #       where dt = 1/spd, so dt*segsize=[length of segment in time]
    #       and dlon = lon[1]-lon[0] (= size of longitude dimension in degrees)
    #       _When only the positive frequencies are used, also multiply by 2._
    # AND: the integral of the power spectral density is then equal to the variance
    #      In this case, gets the variance of x_wintap; for suitably large segsize,
    #      the tapering shouldn't matter much, so VAR[x_wintap] ≃ VAR[x_win]
    # TEST OF INT[psd] = VAR[x_wintap]:
    # dlon = x_wintap['lon'][1].item()-x_wintap['lon'][0].item()
    # dt = 1/spd
    # Nlon = lon_size
    # Nt = segsize
    # print(f"{dlon = }, {dt = }, {Nlon = }, {Nt = }")
    # psd = (np.abs(z)**2) * dlon * dt * (Nlon * Nt)
    # logging.debug(f"{psd.shape = }")
    # variance = np.var(x_wintap, axis=(2,3))
    # logging.debug(f"VARIANCE ARRAY IS LEFT AS: {variance.shape = }")
    # kx = np.fft.fftfreq(Nlon, dlon)
    # ky = np.fft.fftfreq(Nt, dt)
    # psd_integral = np.sum(psd, axis=(2,3))* (kx[1]-kx[0]) * (ky[1]-ky[0])
    # logging.debug(f"PSD_INTEGRAL SHAPE: {psd_integral.shape}")
    # print(f"Variance of data (0,0): {variance[0,0]} -- Integral of spectrum: {psd_integral[0,0]}")

    # z has both positive & negative frequencies : usually you'd take the positive in each dimension and double it

    # Or do the transform with 2 steps (equivalent!)
    # z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
    # z = np.fft.fft(z, axis=3) / segsize

    z = xr.DataArray(
        z,
        dims=("time", "lat", "wavenumber", "frequency"),
        coords={
            "time": x_wintap["time"],
            "lat": x_wintap["lat"],
            "wavenumber": np.fft.fftfreq(lon_size, 1 / lon_size),
            "frequency": np.fft.fftfreq(segsize, 1 / spd),
        },
    )

    # The FFT is returned following ``standard order`` which has negative frequencies in second half of array.
    #
    # IMPORTANT:
    # If this were typical 2D FFT, we would do the following to get the frequencies and reorder:
    #         z_k = np.fft.fftfreq(x_wintap.shape[-2], 1/lon_size)
    #         z_v = np.fft.fftfreq(x_wintap.shape[-1], 1)  # Assumes 1/(1-day) timestep
    # reshape to get the frequencies centered
    #         z_centered = np.fft.fftshift(z, axes=(2,3))
    #         z_k_c = np.fft.fftshift(z_k)
    #         z_v_c = np.fft.fftshift(z_v)
    # and convert to DataArray as this:
    #         d1 = list(x_win.dims)
    #         d1[-2] = "wavenumber"
    #         d1[-1] = "frequency"
    #         c1 = {}
    #         for d in d1:
    #             if d in x_win.coords:
    #                 c1[d] = x_win[d]
    #             elif d == "wavenumber":
    #                 c1[d] = z_k_c
    #             elif d == "frequency":
    #                 c1[d] = z_v_c
    #         z_centered = xr.DataArray(z_centered, dims=d1, coords=c1)
    # BUT THAT IS INCORRECT TO GET THE PROPAGATION DIRECTION OF ZONAL WAVES
    # (in testing, it seems to end up opposite in wavenumber)
    # Apply reordering per Hayashi to get correct wave propagation convention
    #     this function is customized to expect z to be a DataArray
    z_pee = resolveWavesHayashi(z, segsize // spd, spd)
    # z_pee is spectral power already.
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if dosymmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_symmetric = (
            2.0
            * z_pee.isel(lat=z_pee.lat < 0).mean(dim="time")
        )
        z_symmetric = _apply_lat_aggregation(z_symmetric, lataggreg)
        z_symmetric.name = "power"
        z_antisymmetric = (
            2.0
            * z_pee.isel(lat=z_pee.lat > 0).mean(dim="time")
        )
        z_antisymmetric = _apply_lat_aggregation(z_antisymmetric, lataggreg)
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords({"component": ["symmetric", "antisymmetric"]})
    else:
        lat = z_pee["lat"]
        lat_inds = np.argwhere(((lat <= nlat) & (lat >= slat)).values).squeeze()
        z_final = z_pee.isel(lat=lat_inds).mean(dim="time")
        z_final = _apply_lat_aggregation(z_final, lataggreg)
    return z_final

def _apply_lat_aggregation(d, lataggreg):
    if lataggreg == 'sum':
        r = d.sum(dim='lat').squeeze()
    elif lataggreg == 'mean':
        r = d.mean(dim='lat').squeeze()
    else:
        raise ValueError(f"lataggreg set to {lataggreg}, must be `mean` or `sum`")
    return r


def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)

    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe)  # this was an input originally, but I don't know why.
    pi = np.pi
    radius = 6.37122e06  # [m]   average radius of earth
    g = 9.80665  # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05  # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll = 2.0 * pi * radius * np.cos(np.abs(rlat))
    Beta = 2.0 * omega * np.cos(np.abs(rlat)) / radius
    fillval = 1e20

    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType + 1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed
            L = np.sqrt(
                c / Beta
            )  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation

            for wn in range(1, nPlanetaryWave + 1):
                s = -20.0 * (wn - 1) * 2.0 / (nPlanetaryWave - 1) + 20.0
                k = 2.0 * np.pi * s / ll
                kn = k * L

                # Anti-symmetric curves
                if ww == 1:  # MRG wave
                    if k < 0:
                        dell = np.sqrt(1.0 + (4.0 * Beta) / (k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)

                    if k == 0:
                        deif = np.sqrt(c * Beta)

                    if k > 0:
                        deif = fillval

                if ww == 2:  # n=0 IG wave
                    if k < 0:
                        deif = fillval

                    if k == 0:
                        deif = np.sqrt(c * Beta)

                    if k > 0:
                        dell = np.sqrt(1.0 + (4.0 * Beta) / (k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)

                if ww == 3:  # n=2 IG wave
                    n = 2.0
                    dell = Beta * c
                    deif = np.sqrt((2.0 * n + 1.0) * dell + (g * he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1, 5 + 1):
                        deif = np.sqrt(
                            (2.0 * n + 1.0) * dell
                            + (g * he) * k**2
                            + g * he * Beta * k / deif
                        )

                # symmetric curves
                if ww == 4:  # n=1 ER wave
                    n = 1.0
                    if k < 0.0:
                        dell = (Beta / c) * (2.0 * n + 1.0)
                        deif = -Beta * k / (k**2 + dell)
                    else:
                        deif = fillval

                if ww == 5:  # Kelvin wave
                    deif = k * c

                if ww == 6:  # n=1 IG wave
                    n = 1.0
                    dell = Beta * c
                    deif = np.sqrt((2.0 * n + 1.0) * dell + (g * he) * k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1, 5 + 1):
                        deif = np.sqrt(
                            (2.0 * n + 1.0) * dell
                            + (g * he) * k**2
                            + g * he * Beta * k / deif
                        )

                eif = deif  # + k*U since  U=0.0
                P = 2.0 * np.pi / (eif * 24.0 * 60.0 * 60.0)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.

                Apzwn[ww - 1, ed - 1, wn - 1] = s
                if deif != fillval:
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww - 1, ed - 1, wn - 1] = 1.0 / P
                else:
                    Afreq[ww - 1, ed - 1, wn - 1] = fillval
    return Afreq, Apzwn


def kf_filter(data):
    """
    Follows Wheeler-Kiladis and replicates NCL's kf_filter.
    Uses the entire time instead of breaking into segments.

    data: xr.DataArray
        NCL VERSION USES ONLY (time, lon)
        so we to that here, too

    """
    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    xmean = data.mean(dim="time")
    time_axis_index = data.dims.index("time")
    xdetr = xr.DataArray(
        detrend(data.values, axis=time_axis_index, type="linear"),
        dims=data.dims,
        coords=data.coords,
    )
    # xdetr += xmean # put the mean back in (not included in NCL kf_filter)

    # taper to the mean
    xtapr = split_hann_taper(series_length, fraction)
    xtapr = (xdetr - xdetr.mean(dim="time")) * xtapr + xdetr.mean(dim="time")

    # fft
    z = np.fft.fft2(xtapr)


def get_params(lat, radius=6.37122e06, omega=7.292e-05):
    """Returns latitude in radians,
    the perimeter of the small-circle at latitude,
    beta factor at latitude

    requires planetary radius (meters) and angular velocity (radians/second),
    which have default values that are appropriate for Earth.
    """
    pi = np.pi
    latr = np.radians(np.absolute(lat))
    perimeter = 2.0 * pi * radius * np.cos(latr)  # meters around latitude
    beta = 2.0 * omega * np.cos(latr) / radius
    return latr, perimeter, beta


def get_dispersion_curves(
    wavenumber_dim, dispersion_relation, lat, min_edepth, max_edepth
):
    """Convenience function to get the dispersion curve values (frequency)
    for wavenumbers in wavenumber_dim bounded by min/max equivalent depths.

     wavenumber_dim
         array of wavenumbers, determines the output size

     dispersion_relation : func
         function to call to calculate the dispersion relation

     lat : float
         latitude (degrees)

     min_edepth, max_edepth : float
         minimum and maximum equivalent depth (meters)

     returns f1, f2
         arrays of frequencies at each wavenumber


    """
    wavenumber_array = xr.DataArray(wavenumber_dim, dims=["wavenumber"])
    # Calculate f1 and f2 in a vectorized manner
    f1 = xr.apply_ufunc(
        dispersion_relation,
        lat,
        wavenumber_array,
        min_edepth,
        input_core_dims=[[], ["wavenumber"], []],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["wavenumber"]],
    )

    f2 = xr.apply_ufunc(
        dispersion_relation,
        lat,
        wavenumber_array,
        max_edepth,
        input_core_dims=[[], ["wavenumber"], []],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["wavenumber"]],
    )
    return f1, f2


def kelvin_wave_dispersion(latitude, wavenumber, equivalent_depth):
    """Given wavenumber get the Kelvin mode frequency (cpd)
       for the given latitude and equivalent depth.

    latitude : float
      latitude value (scalar) in degrees

    wavenumber : array-like
      all the wavenumbers for which to get the frequencies
    """
    latr, perimeter, beta = get_params(latitude)
    # When wavenumber is an array, k & angular_frequency are arrays
    k = 2.0 * np.pi * wavenumber / perimeter  # wavenumber in [rad m^{-1}]
    angular_frequency = np.sqrt(g * equivalent_depth) * k  # [rad s^{-1}]
    with np.errstate(divide="ignore", invalid="ignore"):
        period = (2.0 * np.pi) / (angular_frequency * 86400.0)
    frequency = 1 / period  # cycles per day
    return frequency


def eq_rossby_wave_dispersion(latitude, wavenumber, equivalent_depth):
    """Given wavenumber get the (n=1) Equatorial rossby mode frequency (cpd)
    for the given latitude and equivalent depth.

    note: In Gehne's code, there is a solver used to solve
    for the roots of another dispersion relation.
    Not sure if that is b/c they use the primitive equations
    versus the SWE. Their ref is Wheeler & Nguyen (eq 13)
    """
    n = 1.0  # This should probably actually be an integer
    latr, perimeter, beta = get_params(latitude)
    k = 2.0 * np.pi * wavenumber / perimeter  # wavenumber in [rad m^{-1}]
    c = np.sqrt(g * equivalent_depth)  # phase speed, m/s
    # if k < 0:
    #     angular_frequency = (-1*beta*k)/(k*k + ((2*n +1)*beta)/c) # [rad s^{-1}]
    #     period = (2.*pi)/(angular_frequency * 86400.)
    #     frequency = 1 /  period # cycles per day
    #     return frequency
    # else:
    #     return np.nan
    angular_frequency = (-1 * beta * k) / (
        k * k + ((2 * n + 1) * beta) / c
    )  # [rad s^{-1}]
    period = (2.0 * np.pi) / (angular_frequency * 86400.0)
    frequency = 1 / period  # cycles per day
    return xr.where(k < 0, frequency, np.nan)


def mrg_wave_dispersion(latitude, wavenumber, equivalent_depth):
    # note: the dispersion is nearly identical to eig0
    # because they are the positive and negative roots for the n=0
    # solution.
    latr, perimeter, beta = get_params(latitude)
    k = 2.0 * np.pi * wavenumber / perimeter  # wavenumber in [rad m^{-1}]
    c = np.sqrt(g * equivalent_depth)
    angular_frequency = 0.5 * c * k * (1 - np.sqrt(1 + (4 * beta / (k**2 * c))))
    # ..............................^................................
    # change to + for the eastward IG0 wave.
    period = (2 * np.pi) / (angular_frequency * 86400)
    frequency = 1 / period
    # print(f"mrg_wave_dispersion: {c = }, min k {k.min().item()}, max k {k.max().item()}, min w {angular_frequency.min().item()}, max w {angular_frequency.max().item()}")
    return frequency  # xr.where(k<0,frequency, np.nan)


def kelvin_wave_mask(wavenumber_dim, frequency_dim, do_profiling=None):
    # define the function that will be used
    # for the dispersion curves
    dispersion_relation = kelvin_wave_dispersion
    # Create meshgrid of wavenumber and frequency
    with optional_timer(do_profiling, "[kelvin_wave_mask] broadcast"):
        wn, freq = xr.broadcast(wavenumber_dim, frequency_dim)

    # KELVIN REGION:
    # Hard coded: Equivalent depths 8m & 90m
    lat = 0
    min_period = 2.5  # days
    max_period = 20  # days
    max_wavenum = 14
    min_wavenum = 1
    min_edepth = 8
    max_edepth = 90

    # Frequency cutoffs: 17 and 2.5 cycles-per-day
    # return frequencies for each wavenumber
    # f1 = []
    # f2 = []
    # for k in wavenumber_dim:
    #     f1.append(kelvin_wave_dispersion(lat, k, min_edepth))
    #     f2.append(kelvin_wave_dispersion(lat, k, max_edepth))
    # f1 = xr.DataArray(f1, dims=['wavenumber'], coords={'wavenumber': wavenumber_dim}).broadcast_like(freq)
    # f2 = xr.DataArray(f2, dims=['wavenumber'], coords={'wavenumber': wavenumber_dim}).broadcast_like(freq)

    with optional_timer(
        do_profiling, "[kelvin_wave_mask] dispersion curves, broadcast 2"
    ):
        f1, f2 = get_dispersion_curves(
            wavenumber_dim, dispersion_relation, lat, min_edepth, max_edepth
        )
        # Broadcast f1 and f2 to match freq dimensions
        f1 = f1.broadcast_like(freq)
        f2 = f2.broadcast_like(freq)

    # Create masks
    with optional_timer(do_profiling, "[kelvin_wave_mask] masking"):
        # Frequency/Period limits; not really needed for KW b/c dispersion bounds frequency
        mask_freq = xr.where(
            (np.abs(freq) >= 1 / max_period) & (np.abs(freq) <= 1 / min_period), 1, 0
        )
        # Dispersion Curve limits
        mask_above_line = wn.copy().astype(int)
        # Assuming wavenumber_dim, freq, f1, and f2 are xarray DataArrays
        wav_sign = xr.where(wavenumber_dim < 0, -1, 1)
        # Create masks for positive and negative wavenumbers
        mask_positive = (freq > f1) & (freq < f2)
        mask_negative = (freq < f1) & (freq > f2)
        # Combine masks based on wavenumber sign
        mask_above_line = xr.where(wav_sign < 0, mask_negative, mask_positive)
        # Convert boolean to int (1 and 0) (not strictly necessary, but to be consistent)
        mask_above_line = mask_above_line.astype(int)
        # Wavenumber limits
        mask_wavenumber = xr.where(
            (np.abs(wn) >= min_wavenum) & (np.abs(wn) <= max_wavenum), 1, 0
        )

    # Combine masks
    final_mask = mask_wavenumber & mask_freq & mask_above_line
    return final_mask


def equatorial_rossby_wave_mask(wavenumber_dim, frequency_dim):
    dispersion_relation = eq_rossby_wave_dispersion
    lat = 0
    wn, freq = xr.broadcast(wavenumber_dim, frequency_dim)
    # f1 = []
    # f2 = []
    min_edepth = 8  # m
    max_edepth = 90
    min_wavenum = 1
    max_wavenum = 10
    # for k in wavenumber_dim:
    #     f1.append(eq_rossby_wave_dispersion(lat, k, min_edepth))
    #     f2.append(eq_rossby_wave_dispersion(lat, k, max_edepth))
    # f1 = np.array(f1)
    # f2 = np.array(f2)

    f1, f2 = get_dispersion_curves(
        wavenumber_dim, dispersion_relation, lat, min_edepth, max_edepth
    )
    # Broadcast f1 and f2 to match freq dimensions
    f1 = f1.broadcast_like(freq)
    f2 = f2.broadcast_like(freq)

    # Create masks
    mask_freq = xr.where((np.abs(freq) >= 0), 1, 0)

    mask_wavenumber = xr.where(
        (np.abs(wn) >= min_wavenum) & (np.abs(wn) <= max_wavenum), 1, 0
    )

    # mask_above_line = wn.copy().astype(int)

    # for i, wav in enumerate(wavenumber_dim):
    #     if wav > 0:
    #         mask_above_line[i,:] = xr.where( (freq[i,:] < f1[i])&(freq[i,:] > f2[i]), 1, 0)
    #     else:
    #         mask_above_line[i,:] = xr.where( (freq[i,:] > f1[i])&(freq[i,:] < f2[i]), 1, 0)

    ## Vectorize
    # Dispersion Curve limits
    mask_above_line = wn.copy().astype(int)
    # Assuming wavenumber_dim, freq, f1, and f2 are xarray DataArrays
    wav_sign = xr.where(wavenumber_dim < 0, -1, 1)
    # Create masks for positive and negative wavenumbers
    mask_positive = (freq < f1) & (freq > f2)
    mask_negative = (freq > f1) & (freq < f2)
    # Combine masks based on wavenumber sign
    mask_above_line = xr.where(wav_sign < 0, mask_negative, mask_positive)
    # Convert boolean to int (1 and 0) (not strictly necessary, but to be consistent)
    mask_above_line = mask_above_line.astype(int)
    ##

    # mask_above_ten = xr.where(np.abs(wn)>10, 0, 1)
    # Combine masks
    final_mask = mask_freq & mask_above_line & mask_wavenumber
    return final_mask


def mrg_wave_mask(wavenumber_dim, frequency_dim):
    dispersion_relation = mrg_wave_dispersion
    lat = 0
    wn, freq = xr.broadcast(wavenumber_dim, frequency_dim)
    min_edepth = 8  # m
    max_edepth = 90
    min_wavenum = 1
    max_wavenum = 10
    f1, f2 = get_dispersion_curves(
        wavenumber_dim, dispersion_relation, lat, min_edepth, max_edepth
    )
    # Broadcast f1 and f2 to match freq dimensions
    f1 = f1.broadcast_like(freq)
    f2 = f2.broadcast_like(freq)
    # Create masks
    mask_wavenumber = xr.where(
        (np.abs(wn) >= min_wavenum) & (np.abs(wn) <= max_wavenum), 1, 0
    )
    # Dispersion Curve limits
    mask_above_line = wn.copy().astype(int)
    # Assuming wavenumber_dim, freq, f1, and f2 are xarray DataArrays
    wav_sign = xr.where(wavenumber_dim < 0, -1, 1)
    # Create masks for positive and negative wavenumbers
    mask_positive = (freq < f1) & (freq > f2)
    mask_negative = (freq > f1) & (freq < f2)
    # Combine masks based on wavenumber sign
    mask_above_line = xr.where(wav_sign < 0, mask_negative, mask_positive)
    # Convert boolean to int (1 and 0) (not strictly necessary, but to be consistent)
    mask_above_line = mask_above_line.astype(int)
    # Combine masks
    final_mask = mask_above_line & mask_wavenumber
    return final_mask


def wv_block_wave_mask(
    wavenumber_dim, frequency_dim, min_wav, max_wav, min_freq, max_freq
):
    wn, freq = xr.broadcast(wavenumber_dim, frequency_dim)
    mask = xr.where((wn >= min_wav) & (wn <= max_wav), 1, 0)
    mask = xr.where((freq >= min_freq) & (freq <= max_freq), mask, 0)
    return mask
