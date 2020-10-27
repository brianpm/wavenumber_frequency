# `wavenumber_frequency`

A pretty straightforward implementation of the wavenumber-frequency spectral analysis following Wheeler & Kiladis.

The approach closely follows the NCL implementation. The main difference is that the filtering rule is simpler here. 

The file `wavenumber_frequency_functions.py` contains all the functions needed to produce "wheeler-kiladis diagrams."

The file `example_analysis_script.py` shows the minimal steps needed to run the analysis and plot the result. I have omitted the data preparation, but it could be as simple as using `x = xarray.open_dataset()['precip']`. 

There are lots of notes in the first file, including some directly copied from NCL's source code. 

At this point, I believe this approach produces results that are _very_ similar to NCL. This has only been tested, however, using daily mean precipitation from climate model output. Missing/invalid data may not be handled correctly. Other variables should be fine because I do not think there are any hard-coded values or assumptions about the physical units, but I haven't tested.