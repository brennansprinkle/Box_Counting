# Box_Counting
Some codes to count particles in boxes and calculate statisics

To install, clone this directory into your site-packages directory (eg `cd ~/.local/lib/python3.10/site-packages && git clone https://github.com/brennansprinkle/Box_Counting countoscope`)

To use
```py
import countoscope.Numba_Box_Count_Stats as countoscope

N2_mean, N2_std, N_stats = countoscope.Calc_and_Output_Stats(data=f"data.dat", 
                                                             Nframes=2400, 
                                                             window_size_x=217.6, window_size_y=174, 
                                                             box_sizes=Box_Ls, sep_sizes=sep)
```
See the full example in `example.py`.

The parameters to `Calc_and_Output_Stats` are:
* `data`. Either:
  * `data` should be a string, the address of a text file containing rows of x, y, t values, whitespace separated. 
  * or `data` should be provided as a 2D array where `data[i, 0:2]` is `[x, y, t]`
* `Nframes` optional if `data` is provided as an address to a file. If not supplied, we will find it, which adds another iteration of the input array when the input data is supplied in a file. Not needed when `data` is an array.
* `window_size_x` and `window_size_y`
* The box sizes are specified as:
  * if only `box_sizes` is provided, the boxes will be square, width and height of box `i` equal to `box_sizes[i]`.
  * if `box_sizes_x` and `box_sizes_y` are provided, box `i` will be of shape `box_sizes_x[i]` * `box_sizes_y[i]`.
  * however, if you want one of the width or height to be constant, you can just pass a single value to one of `box_sizes_x` or `box_sizes_y` and then all boxes will have the same width or height.
* `sep_sizes` should be an array of the same size as `box_sizes`/`box_sizes_x`/`box_sizes_y`-

The return values:
* `N2_mean` and `N2_std` are arrays of shape (len(box_sizes) x Nframes).
* `N_stats` is an array of shape (len(box_sizes) x 5) where each row is box size, particle number mean, particle number variance, particle number variance sem_lb(?), particle number variance sem_ub(?)

# C++ box counting
To run the code with a linux OS, first navigate to the working directory in a treminal and compile the C++ module with `make`. If you're running the code on MAC_OS, then rename the file `Makefile` to `Makefile_Linux` and the file `Makefile_MACOS` to `Makefile`.
The C++ module has dependencies:
* pybind11 (install using eg `pip install pybind11`)
* Eigen (download from https://eigen.tuxfamily.org/ and add to include path)
* FFTW (download from https://www.fftw.org/download.html and add to include path)

# Pure python box counting
To run the pure python code, simply modify the main `Fast_Box_Stats_NoCpp.py` and run with `python Fast_Box_Stats_NoCpp.py`

# Timescale integral
The MATLAB file `timescale_integral.m` processes the data computed using `Fast_Box_Stats.py` by computing the timescale integral.

