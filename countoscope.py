import numpy as np
import scipy.stats as stats
import numba
import numba.typed
import warnings

###############################
# These two function are from SE
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
##############################

def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * np.conjugate(F)
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A

@numba.njit(fastmath=True)
def msd_fft1d(r):
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    with numba.objmode(S2='float64[:]'):
        S2 = autocorrFFT(r)
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in numba.prange(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2 * S2

@numba.njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    # calculates the MSDs over axis 2
    MSDs = np.zeros_like(matrix)
    for i in numba.prange(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # these two loops will be slightly slower than just having one loop
            # that iterates through all combinations of i and j in one go,
            # due to not fully utilising the parallel capability
            MSDs[i, j, :] = msd_fft1d(matrix[i, j, :])
    return MSDs

@numba.njit(parallel=True, fastmath=True)
def msd_coords(Xs,Ys):
    numRows, numCols = Xs.shape
    MSDs = np.zeros((numRows, numCols))

    for i in numba.prange(numRows):
        xi = Xs[i, :]
        yi = Ys[i, :]
        row_i = msd_fft1d(xi) + msd_fft1d(yi)
        MSDs[i,:] = row_i
    
    return MSDs

def processDataFile(filename):
    data = np.fromfile(filename, dtype=float, sep=' ')
    #all_data = np.loadtxt('data/0.34_EKRM_trajs.dat', delimiter=',', skiprows=1)
    data = data.reshape((-1, 4))
    
    return processDataArray(data)

@numba.njit()
def processDataArray(data):
    # returns (Xs, Ys) where Xs, Ys are lists, and Xs[t]/Ys[t] is a list of the x/y coordinates at time tassert data[:, 2].min() == 1, f'data timesteps should (presently) be 1-based. The first timestep was {data[:, 2].min()}'
    t0 = int(data[:, 2].min())
    Nframes = int(data[:, 2].max()) + 1 - t0 # this works because max_t is actually max(t)+1, and Nframes should be max(t)+1 when zero based

    # storing the data by continually appending to lists is incredibly slow
    # so instead we use a numpy array
    # but first we need to find the maximum number of simultaneous particles
    num_points_at_time = np.zeros((Nframes), dtype='int')
    for line_i in range(data.shape[0]):
        values = data[line_i, :]
        t = round(values[2])
        num_points_at_time[t-t0] += 1
 
    # then we add the particle coordinates into the numpy arrays
    Xs_ = np.full((Nframes, num_points_at_time.max()), np.nan)
    Ys_ = np.full((Nframes, num_points_at_time.max()), np.nan)
    num_points_at_time = np.zeros((Nframes), dtype='int')

    for line_i in range(data.shape[0]):
        values = data[line_i, :]
        
        x = values[0]
        y = values[1]
        t = round(values[2])

        p = num_points_at_time[t-t0]
        Xs_[t-t0, p] = x
        Ys_[t-t0, p] = y

        num_points_at_time[t-t0] += 1

    # finally we convert them back to python lists
    # perhaps we could skip this step in the future
    Xs = []
    Ys = []
    for t_index in range(Nframes):
        Xs.append(list(Xs_[t_index, 0:num_points_at_time[t_index]]))
        Ys.append(list(Ys_[t_index, 0:num_points_at_time[t_index]]))

    min_x = data[:, 0].min()
    max_x = data[:, 0].max()
    min_y = data[:, 1].min()
    max_y = data[:, 1].max()
    return Xs, Ys, min_x, max_x, min_y, max_y

@numba.njit
def do_counting_at_boxsize(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, offset_x, offset_y, box_index, num_timesteps, SepSize_x, SepSize_y):
    
    num_boxes_x = int(np.floor(window_size_x / SepSize_x))
    num_boxes_y = int(np.floor(window_size_y / SepSize_y))

    counts = np.zeros((num_boxes_x, num_boxes_y, num_timesteps), dtype=np.float32) # why do we specify the dtype here?
    
    num_boxes_x = int(np.floor(window_size_x / SepSize_x))
    num_boxes_y = int(np.floor(window_size_y / SepSize_y))
    assert num_boxes_x > 0, "Nx was zero"
    assert num_boxes_y > 0, "Ny was zero"
    
    for time_index in numba.prange(num_timesteps):
        xt = x[time_index]
        yt = y[time_index]
        num_points = len(xt) # number of x,y points available at this timestep

        for i in range(num_points):
            # find target box
            if xt[i] - offset_x[box_index] < 0 or yt[i] - offset_y[box_index] < 0:
                continue # these are points close to the origin that fall before the first box, when there's an offset
            target_box_x = int(np.floor((xt[i] - offset_x[box_index]) / SepSize_x))
            target_box_y = int(np.floor((yt[i] - offset_y[box_index]) / SepSize_y))

            # if the target box doesn't entirely fit within the window, discard the point
            if (target_box_x+1.0) * SepSize_x > window_size_x:
                continue
            if (target_box_y+1.0) * SepSize_y > window_size_y:
                continue

            # discard points that are in the sep border around the edge of the box
            distance_into_box_x = np.fmod(xt[i], SepSize_x)
            distance_into_box_y = np.fmod(yt[i], SepSize_y)
            if np.abs(distance_into_box_x-0.5*SepSize_x) >= box_sizes_x[box_index]/2.0:
                continue
            if np.abs(distance_into_box_y-0.5*SepSize_y) >= box_sizes_y[box_index]/2.0:
                continue
                        
            # add this particle to the stats
            counts[target_box_x, target_box_y, time_index] += 1.0
            
    return counts

@numba.njit(fastmath=True)
def count_boxes(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, sep_sizes, offset_x=None, offset_y=None):
    # offset_x and offset_y will offset the whole grid of boxes from the origin
    # TODO: if offset is so big to reduce the number of boxes we might have a problem

    if offset_x == None:
        offset_x = np.zeros_like(box_sizes_x)
    if offset_y == None:
        offset_y = np.zeros_like(box_sizes_y)
    assert offset_x.shape == box_sizes_x.shape
    assert offset_y.shape == box_sizes_y.shape

    CountMs = numba.typed.List()
    # each of the arrays that gets appended in here will have a different shape
    # which seems to be why we use a list not an ndarray

    for box_index in range(len(box_sizes_x)):
        num_timesteps = len(x)
        
        overlap = sep_sizes[box_index] < 0 # do the boxes overlap?
        
        print("Counting boxes L =", box_sizes_x[box_index], "*", box_sizes_y[box_index], ", sep =", sep_sizes[box_index], ", num =", "overlapped" if overlap else "")

        if overlap and (box_sizes_x[box_index] % np.abs(sep_sizes[box_index]) == 0 or box_sizes_y[box_index] % np.abs(sep_sizes[box_index]) == 0):
            print('Negative overlap is an exact divisor of box size. This will lead to correlated boxes.')
        
        assert num_timesteps > 0
        
        if overlap:

            x_shift_size = box_sizes_x[box_index] + sep_sizes[box_index] # remember sep is negative in this regime
            y_shift_size = box_sizes_y[box_index] + sep_sizes[box_index] # remember sep is negative in this regime

            num_x_shifts = int(box_sizes_x[box_index] // x_shift_size)
            num_y_shifts = int(box_sizes_y[box_index] // y_shift_size)

            SepSize_x = num_x_shifts * x_shift_size # SepSize is (in non overlapped sense) L+sep, needs a proper namev
            SepSize_y = num_y_shifts * y_shift_size

            total_num_boxes_x = int(np.floor(window_size_x / SepSize_x) * num_x_shifts)
            total_num_boxes_y = int(np.floor(window_size_y / SepSize_y) * num_y_shifts)
            Counts = np.zeros((total_num_boxes_x, total_num_boxes_y, num_timesteps), dtype=np.float32)

            for x_shift_index in range(num_x_shifts):
                x_shift = x_shift_size * x_shift_index

                this_offset_x = offset_x + x_shift

                for y_shift_index in range(num_y_shifts):
                    y_shift = y_shift_size * y_shift_index
                    this_offset_y = offset_y + y_shift

                    Counts[x_shift_index::num_x_shifts, y_shift_index::num_y_shifts, :] = do_counting_at_boxsize(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, this_offset_x, this_offset_y, box_index, num_timesteps, SepSize_x, SepSize_y)
                    # now we have the counts, we have to kind of "inter-tile" them to get the (num boxes x) * (num_boxes y) * (num timesteps) shape in a sensible way

        else:
            SepSize_x = box_sizes_x[box_index] + sep_sizes[box_index] # SepSize is (in non overlapped sense) L+sep, needs a proper name
            SepSize_y = box_sizes_y[box_index] + sep_sizes[box_index]

            Counts = do_counting_at_boxsize(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, offset_x, offset_y, box_index, num_timesteps, SepSize_x, SepSize_y)

        CountMs.append(Counts)

    print("Done with counting")
    return CountMs

@numba.njit(fastmath=True)
# this jitting is probably pointless, no?
def computeMeanAndSecondMoment(matrix):
    return np.mean(matrix), np.var(matrix)

def check_provided_box_sizes(box_sizes, box_sizes_x, box_sizes_y, sep_sizes):
    if box_sizes is not None:
        assert box_sizes_x is None and box_sizes_y is None, "if parameter box_sizes is provided, neither box_sizes_x nor box_sizes_y should be provided"
        box_sizes_x = box_sizes
        box_sizes_y = box_sizes
    else:
        assert box_sizes_x is not None and box_sizes_y is not None, "if box_sizes is not provided, both box_sizes_x and box_sizes_y should be provided"

        if np.isscalar(box_sizes_x):
            assert not np.isscalar(box_sizes_y), "if box_sizes_x is provided as a scalar, box_sizes_y should be an array"
            box_sizes_x = np.full_like(box_sizes_y, box_sizes_x)
        elif np.isscalar(box_sizes_y):
            assert not np.isscalar(box_sizes_x), "if box_sizes_y is provided as a scalar, box_sizes_x should be an array"
            box_sizes_y = np.full_like(box_sizes_x, box_sizes_y)
        
        assert len(box_sizes_x) == len(box_sizes_y)
        
    box_sizes_x = np.array(box_sizes_x) # ensure these are numpy arrays, not python lists or tuples
    box_sizes_y = np.array(box_sizes_y)
    
    assert np.all(~np.isnan(box_sizes_x)), "nan was found in box_sizes_x"
    assert np.all(~np.isnan(box_sizes_y)), "nan was found in box_sizes_y"
    assert np.all(~np.isnan(sep_sizes)),   "nan was found in sep_sizes"
    
    if np.isscalar(sep_sizes):
        sep_sizes = np.full_like(box_sizes_x, sep_sizes)
    else:
        assert len(box_sizes_x) == len(sep_sizes), "box_sizes(_x) and sep_sizes should have the same length"

    sep_sizes   = np.array(sep_sizes)
    assert np.all(- sep_sizes < box_sizes_x), '(-1) * sep_sizes[i] must always be smaller than box_sizes[i]'
    assert np.all(- sep_sizes < box_sizes_y)

    return box_sizes_x, box_sizes_y, sep_sizes

def load_data_and_check_window_size(data, window_size_x, window_size_y):
    # load data
    if type(data) is str:
        print("Reading data from file")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataFile(data)
    else:
        assert np.all(~np.isnan(data)), "nan was found in data"
        print("Reading data from array")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataArray(data)
    print("Done with data read")

    # check the window size is sensible
    if window_size_x is None:
        window_size_x = max_x
        print(f'Assuming window_size_x={window_size_x:.1f}')
    if window_size_y is None:
        window_size_y = max_y
        print(f'Assuming window_size_y={window_size_y:.1f}')
    # TODO: we haven't done anything if min_x is not zero

    assert min_x >= 0,             'An x-coordinate was supplied less than zero'
    assert max_x <= window_size_x, 'An x-coordinate was supplied greater than window_size_x'
    assert min_y >= 0,             'A y-coordinate was supplied less than zero'
    assert max_y <= window_size_y, 'A y-coordinate was supplied greater than window_size_x'

    warn_empty_thresh = 0.9
    if (max_x-min_x) < warn_empty_thresh * window_size_x:
        warnings.warn(f'x data fills less than {100*(max_x-min_x)/window_size_x:.0f}% of the window. Is window_size_x correct?')
    if (max_y-min_y) < warn_empty_thresh * window_size_y:
        warnings.warn(f'y data fills less than {100*(max_y-min_y)/window_size_y:.0f}% of the window. Is window_size_y correct?')
    
    for x_list, y_list in zip(Xs, Ys):
        assert np.all(~np.isnan(x_list)), "nan was found in Xs"
        assert np.all(~np.isnan(y_list)), "nan was found in Ys"

    return Xs, Ys, window_size_x, window_size_y

def calculate_nmsd(data, sep_sizes, window_size_x=None, window_size_y=None, box_sizes=None, box_sizes_x=None, box_sizes_y=None):
    # input parameter processing
    box_sizes_x, box_sizes_y, sep_sizes = check_provided_box_sizes(box_sizes, box_sizes_x, box_sizes_y, sep_sizes)
    
    # load the data and check it
    Xs, Ys, window_size_x, window_size_y = load_data_and_check_window_size(data, window_size_x, window_size_y)

    assert np.all(box_sizes_x < window_size_x), "None of box_sizes(_x) can be bigger than window_size_x"
    assert np.all(box_sizes_y < window_size_y), "None of box_sizes(_y) can be bigger than window_size_y"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_x"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_y"

    # now do the actual counting
    print("Compiling fast counting function (this may take a min. or so)")
    Xnb = numba.typed.List(np.array(xi) for xi in Xs) # TODO why is this defined here not inside count_boxes?
    Ynb = numba.typed.List(np.array(yi) for yi in Ys) # TODO why is this defined here not inside count_boxes?

    CountMs = count_boxes(Xnb, Ynb, window_size_x=window_size_x, window_size_y=window_size_y,
                                        box_sizes_x=box_sizes_x, box_sizes_y=box_sizes_y, sep_sizes=sep_sizes)

    N_Stats = np.zeros((len(box_sizes_x), 6))

    MSD_means = np.zeros((len(box_sizes_x), len(Xs)))
    MSD_stds  = np.zeros((len(box_sizes_x), len(Xs)))

    for box_index in range(len(box_sizes_x)):
        # why isn't this a numba.prange?
        print("Processing Box size:", box_sizes_x[box_index], "*", box_sizes_y[box_index])

        N_Stats[box_index, 0] = box_sizes_x[box_index]
        #mean, variance, variance_sem_lb, variance_sem_ub = computeMeanAndSecondMoment(CountMs[lbIdx])
        mean_N, variance = computeMeanAndSecondMoment(CountMs[box_index])

        ####################
        alpha = 0.01
        df = 1.0 * CountMs[box_index].size - 1.0
        chi_lb = stats.chi2.ppf(0.5 * alpha, df)
        chi_ub = stats.chi2.ppf(1.0 - 0.5 * alpha, df)

        variance_sem_lb = (df / chi_lb) * variance
        variance_sem_ub = (df / chi_ub) * variance
        ####################

        N_Stats[box_index, 1] = mean_N
        N_Stats[box_index, 2] = variance
        N_Stats[box_index, 3] = variance_sem_lb
        N_Stats[box_index, 4] = variance_sem_ub
        N_Stats[box_index, 5] = CountMs[box_index].shape[0] * CountMs[box_index].shape[1] # number of boxes counted over

        MSDs = msd_matrix(CountMs[box_index])

        MSD_means[box_index, :] = np.mean(MSDs, axis=(0, 1))
        MSD_stds [box_index, :] = np.std (MSDs, axis=(0, 1))

    return MSD_means, MSD_stds, N_Stats
