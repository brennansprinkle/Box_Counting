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
    # calculates the MSDs of the rows of the provided matrix
    Nrows, Ncols = matrix.shape
    MSDs = np.zeros((Nrows,Ncols))
    for i in numba.prange(Nrows):
        #print(100.0 * ((1.0 * i) / (1.0 * N)), "percent done with MSD calc")
        MSD = msd_fft1d(matrix[i, :])
        MSDs[i,:] = MSD
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


def outputMatrixToFile(matrix, filename):
    # currently unused?
    np.savetxt(filename, matrix, delimiter=' ', fmt='%.10f')
    print("Matrix data has been written to", filename)


def ConvertDataFile(filename):
    # is this function unused?
    fileinput = open(filename, "r")
    if not fileinput:
        print("Error opening file:", filename)
        return

    Ntimes = 0
    x, y, z = [], [], []
    aux1, aux2, aux3, aux4 = 0.0, 0.0, 0.0, 0.0

    # Parse 'filename' to remove the extension (chars after a period) and add the string "_modified.txt" to the result.
    inputFilename = filename
    pos = inputFilename.rfind('.')
    baseFilename = inputFilename[:pos]
    outfile = baseFilename + "_modified.txt"

    fileoutput = open(outfile, "w")
    if not fileoutput:
        print("Error opening output file:", outfile)
        fileinput.close()
        return

    while True:
        line = fileinput.readline().strip()
        if not line:
            break

        parts = int(line)
        Ntimes += 1
        print(Ntimes)

        x = np.zeros(parts)
        y = np.zeros(parts)

        # Read the data directly into the arrays and write to the output file
        for i in range(parts):
            values = fileinput.readline().split()
            x[i] = float(values[0])
            y[i] = float(values[1])
            fileoutput.write("{:.6f} {:.6f} {}\n".format(x[i], y[i], Ntimes))

    fileinput.close()
    fileoutput.close()
    return Ntimes

# def processDataFile(filename, Nframes):
#     Xs = [[] for _ in range(Nframes)]
#     Ys = [[] for _ in range(Nframes)]

#     fileinput = open(filename, "r")
#     if not fileinput:
#         print("Error opening file:", filename)
#         return Xs, Ys

#     ind, ind_p = 0, 0
#     x, y = 0.0, 0.0

#     frame = 0
#     start = 0
#     while True:
#         line = fileinput.readline().strip()
#         if not line:
#             break

#         values = line.split()
#         x = float(values[0])
#         y = float(values[1])
#         ind = int(values[2])

#         if frame == 0 and ind != 0:
#             start = ind
#             frame = 1
#             ind_p = ind - 1
#         if ind_p != ind:
#             print(ind)
        
#         Xs[ind - start].append(x)
#         Ys[ind - start].append(y)
#         ind_p = ind

#     fileinput.close()
#     return Xs, Ys

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

        p = num_points_at_time[t-1]
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


@numba.njit(parallel=True, fastmath=True)
def processDataFile_and_Count(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, sep_sizes):
    CountMs = numba.typed.List()
    for box_index in range(len(box_sizes_x)):
        num_timesteps = len(x)
        SepSize_x = box_sizes_x[box_index] + sep_sizes[box_index]
        SepSize_y = box_sizes_y[box_index] + sep_sizes[box_index]
        num_boxes_x = int(np.floor(window_size_x / SepSize_x))
        num_boxes_y = int(np.floor(window_size_y / SepSize_y))
        
        print("Counting boxes L =", box_sizes_x[box_index], "*", box_sizes_y[box_index], ", sep =", sep_sizes[box_index], ", num =", num_boxes_x, 'x', num_boxes_y)

        overlap = sep_sizes[box_index] < 0 # do the boxes overlap?

        if overlap and (box_sizes_x[box_index] % np.abs(sep_sizes[box_index]) == 0 or box_sizes_y[box_index] % np.abs(sep_sizes[box_index]) == 0):
                print('Negative overlap is an exact divisor of box size. This will lead to correlated boxes.')
        
        assert num_boxes_x > 0, "Nx was zero"
        assert num_boxes_y > 0, "Ny was zero"
        assert num_timesteps > 0, "Times was zero"

        Counts = np.zeros((num_boxes_x * num_boxes_y, num_timesteps), dtype=np.float32)

        if overlap:
            # if the boxes overlap we cannot use the original method (below)
            # so we use this method instead, which is perhaps 25 times slower
            for time_index in numba.prange(num_timesteps):
                xt = x[time_index]
                yt = y[time_index]
                num_points = len(xt) # number of x,y points available at this timestep

                for box_x_index in range(num_boxes_x):
                    for box_y_index in range(num_boxes_y):

                        box_x_min = box_x_index * SepSize_x + sep_sizes[box_index]/2
                        box_x_max = box_x_min + box_sizes_x[box_index]
                        box_y_min = box_y_index * SepSize_y + sep_sizes[box_index]/2
                        box_y_max = box_y_min + box_sizes_y[box_index]

                        for point in range(num_points):
                            if box_x_min < xt[point] and xt[point] <= box_x_max and box_y_min < yt[point] and yt[point] <= box_y_max:
                                Counts[box_x_index * num_boxes_y + box_y_index, time_index] += 1.0

        else:
            for time_index in numba.prange(num_timesteps):
                xt = x[time_index]
                yt = y[time_index]
                num_points = len(xt) # number of x,y points available at this timestep

                for i in range(num_points):
                    # find target box
                    target_box_x = int(np.floor(xt[i] / SepSize_x))
                    target_box_y = int(np.floor(yt[i] / SepSize_y))

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
                    Counts[target_box_x * num_boxes_y + target_box_y, time_index] += 1.0

        CountMs.append(Counts)

    print("Done with counting")
    return CountMs

@numba.njit(fastmath=True)
def computeMeanAndSecondMoment(matrix):
    # calculate the mean and variance of the provided array
    # this function is equivalent to `return np.mean(matrix), np.var(matrix)`
    
    numRows, numCols = matrix.shape
    n = numRows * numCols
    assert n > 0, "numRows * numCols was zero"

    av = 0.0
    m2 = 0.0

    for i in range(numRows):
        for j in range(numCols):
            value = matrix[i, j]
            delta = value - av
            av += delta / (i * numCols + j + 1.0)
            m2 += delta * (value - av)

    variance = m2 / n

    return av, variance

def Calc_and_Output_Stats(data, sep_sizes, window_size_x=None, window_size_y=None, box_sizes=None, box_sizes_x=None, box_sizes_y=None):
    # input parameter processing
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
    
    if np.isscalar(sep_sizes):
        sep_sizes = np.full_like(box_sizes_x, sep_sizes)
    else:
        assert len(box_sizes_x) == len(sep_sizes), "box_sizes(_x) and sep_sizes should have the same length"

    box_sizes_x = np.array(box_sizes_x) # ensure these are numpy arrays, not python lists or tuples
    box_sizes_y = np.array(box_sizes_y)
    sep_sizes   = np.array(sep_sizes)
    
    # load the data and check it
    if type(data) is str:
        print("Reading data from file")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataFile(data)
    else:
        print("Reading data from array")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataArray(data)
    print("Done with data read")

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
    
    assert np.all(box_sizes_x < window_size_x), "None of box_sizes(_x) can be bigger than window_size_x"
    assert np.all(box_sizes_y < window_size_y), "None of box_sizes(_y) can be bigger than window_size_y"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_x"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_y"

    # now do the actual counting
    print("Compiling fast counting function (this may take a min. or so)")
    Xnb = numba.typed.List(np.array(xi) for xi in Xs)
    Ynb = numba.typed.List(np.array(yi) for yi in Ys)
    CountMs = processDataFile_and_Count(Xnb, Ynb, window_size_x=window_size_x, window_size_y=window_size_y,
                                        box_sizes_x=box_sizes_x, box_sizes_y=box_sizes_y, sep_sizes=sep_sizes)

    N_Stats = np.zeros((len(box_sizes_x), 5))

    MSD_means = np.zeros((len(box_sizes_x), len(Xs)))
    MSD_stds  = np.zeros((len(box_sizes_x), len(Xs)))

    for box_index in range(len(box_sizes_x)):
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

        MSDs = msd_matrix(CountMs[box_index])

        MSD_means[box_index, :] = np.mean(MSDs, axis=0)
        MSD_stds [box_index, :] = np.std (MSDs, axis=0)

    return MSD_means, MSD_stds, N_Stats
 

def Calc_MSD_and_Output(infile, outfile, Nframes):
    # currently unused
    #CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep)
    Xs,Ys = processDataFile(infile, Nframes)
    print("Done with data read")
    Xs = np.array(Xs).T
    Ys = np.array(Ys).T
    print("calculating particle MSDs")
    MSDs = msd_coords(Xs,Ys)
    MSDmean = np.mean(MSDs, axis=0)
    MSDsem = np.std(MSDs, axis=0) / np.sqrt(MSDs.shape[0])
    outputMatrixToFile(MSDmean, outfile + "_particles_MSDmean.txt")
    outputMatrixToFile(MSDsem, outfile + "_particles_MSDerror.txt")
