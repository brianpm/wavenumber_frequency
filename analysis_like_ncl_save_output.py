import argparse
from pathlib import Path

import numpy as np
import xarray as xr
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt

def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq
    # save : logical to save output arrays as netcdf
    # ofil : path for output netcdf file (defaults to out.nc)
    do_save = kwargs.pop("save", False)
    ofil = kwargs.pop("ofil", "out.nc")
    
    z2 = wf.spacetime_power(x, **kwargs)
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = wf.smooth_wavefreq(z2avg, kern=wf.simple_smooth_kernel(), nsmooth=50, freq_name='frequency')
    # separate components
    z2_sym = z2[0,...].drop_vars("component")
    z2_asy = z2[1,...].drop_vars("component")
    # normalize
    nspec_sym = z2_sym / background 
    nspec_asy = z2_asy / background
    if do_save:
        print(f"Save is triggered...")
        background.name = "background"
        z2_sym.name = "symmetric"
        z2_asy.name = "antisymmetric"
        dsout = xr.Dataset({"symmetric": z2_sym, "antisymmetric":z2_asy, "background":background})
        dsout.to_netcdf(ofil)
        print(f"Save is complete: {ofil}")        
    return nspec_sym, nspec_asy


def plot_normalized_symmetric_spectrum(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)    
    ax.set_title("Normalized Symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)


def plot_normalized_asymmetric_spectrum(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 1.8, 16), cmap='Spectral_r', extend='both')
    for ii in range(0,3):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)
    ax.set_title("Normalized Anti-symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)

#
# LOAD DATA, x = DataArray(time, lat, lon), e.g., daily mean precipitation
#
def get_data(filename, variablename, hfil=None):
    if Path(filename).is_file():
        try: 
            ds = xr.open_dataset(filename)
        except ValueError:
            ds = xr.open_dataset(filename, decode_times=False)
    elif Path(filename).is_dir():
        assert hfil is not None, "When a directory is provided, must also provide a hfil string to search for files."
        fils = Path(filename).glob(f"*.{hfil}.*.nc")
        if fils:
            ds = xr.open_mfdataset(sorted(fils))
    else:
        print("ERROR get_data unable to figure out what data to load")
        return None
    return ds[variablename]


if __name__ == "__main__":
    #
    # input from arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("ifil")
    parser.add_argument("vname")
    parser.add_argument("--hfil", required=False)
    parser.add_argument("--ofil", required=False)
    parser.add_argument("--case", required=False)
    args = parser.parse_args()

    fili = args.ifil
    vari = args.vname
    #
    # Loading data ... example is very simple
    #
    data = get_data(fili, vari, args.hfil)  # returns OLR
    #
    # Determine sampling (in samples per day)
    #
    spd = (86400 / (86400.*(data.time[1]-data.time[0]).dt.days + (data.time[1]-data.time[0]).dt.seconds)).astype(int).item()
    print(f"Determined samples per day = {spd}")
    #
    # We need to have data in memory:
    if hasattr(data, "compute"):
        data = data.compute()
        print("Moved data into memory.")
    
    #
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (-15,15)  # latitude bounds for analysis
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = -65  # time (days) between temporal windows [segments]
                    # negative means there will be overlapping temporal segments
    twoMonthOverlap = 65
    opt      = {'segsize': nDayWin, 
                'noverlap': twoMonthOverlap, 
                'spd': spd, 
                'latitude_bounds': latBound, 
                'dosymmetries': True, 
                'rmvLowFrq':True,
                'save': True}
    if "ofil" in args:
        print(f"Output file specified: {args.ofil}")
        opt['ofil'] = args.ofil
    # in this example, the smoothing & normalization will happen and use defaults
    symComponent, asymComponent = wf_analysis(data, **opt)
    #
    # Plots ... sort of matching NCL, but not worrying much about customizing.
    #
    if "case" in args:
        casename = args.case
    else:
        casename = "example"
    outPlotName = f"{casename}_{vari}_{args.hfil}_symmetric_plot.png"
    plot_normalized_symmetric_spectrum(symComponent, outPlotName)
    print(f"Output figure 1: {outPlotName}")
    outPlotName = f"{casename}_{vari}_{args.hfil}_asymmetric_plot.png"
    plot_normalized_asymmetric_spectrum(asymComponent, outPlotName)
    print(f"Output figure 2: {outPlotName}")