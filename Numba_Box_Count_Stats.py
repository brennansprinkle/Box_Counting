import numpy as np
import scipy.stats as stats
from numba import jit, njit, prange, objmode
from numba.typed import List as nblist

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

@njit(fastmath=True)
def msd_fft1d(r):
    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    with objmode(S2='float64[:]'):
        S2 = autocorrFFT(r)
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in prange(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2 * S2

@njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    Nrows, Ncols = matrix.shape
    MSDs = np.zeros((Nrows,Ncols))
    for i in prange(Nrows):
        #print(100.0 * ((1.0 * i) / (1.0 * N)), "percent done with MSD calc")
        MSD = msd_fft1d(matrix[i, :])
        MSDs[i,:] = MSD
    return MSDs

@njit(parallel=True, fastmath=True)
def msd_coords(Xs,Ys):
    numRows, numCols = Xs.shape
    MSDs = np.zeros((numRows, numCols))

    for i in prange(numRows):
        xi = Xs[i, :]
        yi = Ys[i, :]
        row_i = msd_fft1d(xi) + msd_fft1d(yi)
        MSDs[i,:] = row_i
    
    return MSDs


def outputMatrixToFile(matrix, filename):
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


def processDataFile(filename, Nframes):
    # returns (Xs, Ys) where Xs, Ys are lists, and Xs[t]/Ys[t] is a list of the x/y coordinates at time t

    with open(filename, "r") as fileinput:
        file_contents = fileinput.readlines()
        
        Xs = [[] for _ in range(Nframes)]
        Ys = [[] for _ in range(Nframes)]

        previous_t = 0
        for line in file_contents:
            values = line.split()
            try:
                x = float(values[0])
                y = float(values[1])
                t = round(float(values[2]))
                if t > Nframes:
                    raise Exception(f"The file had a datapoint with time {t} greater than the supplied Nframes {Nframes}")
                Xs[t-1].append(x)
                Ys[t-1].append(y)
                if previous_t != t: # this is kinda weird, what does this code do? Can the whole thing be done a better way?
                    pass
                previous_t = t
            except (ValueError, IndexError) as err:
                print(f"I can't read index {previous_t} of the file: '{line.strip()}', {err}")
                raise err # this used to be `continue` but for now I see no reason to allow that
    
    return Xs, Ys


@njit(parallel=True, fastmath=True)
def processDataFile_and_Count(x, y, window_size_x, window_size_y, box_sizes, sep_sizes):
    CountMs = nblist()
    for box_index in range(len(box_sizes)):
        print("Counting boxes L =", box_sizes[box_index], ", sep =", sep_sizes[box_index])
        num_timesteps = len(x)
        SepSize = box_sizes[box_index] + sep_sizes[box_index]
        num_boxes_x = int(np.floor(window_size_x / SepSize))
        num_boxes_y = int(np.floor(window_size_y / SepSize))
        
        assert num_boxes_x > 0, "Nx was zero"
        assert num_boxes_y > 0, "Ny was zero"
        assert num_timesteps > 0, "Times was zero"

        Counts = np.zeros((num_boxes_x * num_boxes_y, num_timesteps), dtype=np.float32)

        for time_index in prange(num_timesteps):
            xt = x[time_index]
            yt = y[time_index]
            num_points = len(xt) # number of x,y points available at this time

            for i in range(num_points):
                # periodic corrections
                while xt[i] > window_size_x:
                    xt[i] -= window_size_x
                while xt[i] < 0.0:
                    xt[i] += window_size_x
                while yt[i] > window_size_y:
                    yt[i] -= window_size_y
                while yt[i] < 0.0:
                    yt[i] += window_size_y

                # find correct box and increment counts
                target_box_x = int(np.floor(xt[i] / SepSize))
                target_box_y = int(np.floor(yt[i] / SepSize))

                # if the target box doesn't entirely fit within the window, discard the point
                if (target_box_x+1.0) * SepSize > window_size_x:
                    continue
                if (target_box_y+1.0) * SepSize > window_size_y:
                    continue

                # discard points that are in the sep border around the edge of the box
                distance_into_box_x = np.fmod(xt[i], SepSize)
                distance_into_box_y = np.fmod(yt[i], SepSize)
                if np.abs(distance_into_box_x-0.5*SepSize) >= box_sizes[box_index]/2.0:
                    continue
                if np.abs(distance_into_box_y-0.5*SepSize) >= box_sizes[box_index]/2.0:
                    continue
                    
                # add this particle to the stats
                Counts[target_box_x * num_boxes_y + target_box_y, time_index] += 1.0

        CountMs.append(Counts)

    print("Done with counting")
    return CountMs


@njit(fastmath=True)
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

def Calc_and_Output_Stats(infile, outfile, Nframes, window_size_x, window_size_y, box_sizes, sep_sizes):
    assert len(box_sizes) == len(sep_sizes), "box_sizes and sep_sizes should have the same length"

    #CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep)
    Xs,Ys = processDataFile(infile, Nframes)
    print("Done with data read")
    print("Compiling fast counting function (this may take a min. or so)")
    Xnb = nblist(np.array(xi) for xi in Xs)
    Ynb = nblist(np.array(yi) for yi in Ys)
    CountMs = processDataFile_and_Count(Xnb, Ynb, window_size_x, window_size_y, box_sizes, sep_sizes)

    N_Stats = np.zeros((len(box_sizes), 5))

    for box_index in range(len(box_sizes)):
        print("Processing Box size:", box_sizes[box_index])

        N_Stats[box_index, 0] = box_sizes[box_index]
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

        MSDmean = np.mean(MSDs, axis=0)
        MSDsem = np.std(MSDs, axis=0) / np.sqrt(MSDs.shape[0])
        Lstr = format(box_sizes[box_index], '0.6f')
        outputMatrixToFile(MSDmean, outfile + "_MSDmean_BoxL_" + Lstr + ".txt")
        outputMatrixToFile(MSDsem, outfile + "_MSDerror_BoxL_" + Lstr + ".txt")

    outputMatrixToFile(N_Stats, outfile + "_N_stats.txt")

 

def Calc_MSD_and_Output(infile, outfile, Nframes):
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
