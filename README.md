# Box_Counting
Some codes to count particles in boxes and calculate statisics

To install, copy this directory into your site-packages directory (eg `~/.local/lib/python3.10/site-packages`)

To use
```py
import Box_Counting as countoscope

N2_mean, N2_std, N_stats = countoscope.Calc_and_Output_Stats(infile=f"data.dat", 
                                                             Nframes=2400, 
                                                             window_size_x=217.6, window_size_y=174, 
                                                             box_sizes=Box_Ls, sep_sizes=sep)
```
`data.dat` should be a text file containing rows of x, y, t values, whitespace separated. `N2_mean` and `N2_std` are arrays of shape (len(box_sizes) x Nframes). `N_stats` is an array of shape (len(box_sizes) x 5) where each row is box size, particle number mean, particle number variance, particle number variance sem_lb(?), particle number variance sem_ub(?)

Pass the optional parameter `strip_mode=True` to count particles in vertical strips (of width `box_sizes`) instead of boxes

For compatibility with a previous version of this library, when the output data was written straight to disk, use the following
```py
for i, L in enumerate(N_stats[:, 0]):
    np.savetxt(f`{L}_mean.txt`, N2_mean[i, :], delimiter=' ', fmt='%.10f')
    np.savetxt(f`{L}_std.txt`,  N2_std [i, :], delimiter=' ', fmt='%.10f')
    
np.savetxt(f`N_stats.txt`,  N_stats_, delimiter=' ', fmt='%.10f')
```

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

