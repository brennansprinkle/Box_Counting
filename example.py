import numpy as np
import countoscope.Numba_Box_Count_Stats as countoscope

if __name__ == '__main__':
    Lx = 217.6 # box size x-dir 
    Ly = 174 # box size y-dir
    Box_Ls = np.array([64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]) # array of box sizes to probe

    a = 1.395 #radius of particles
    sep = np.array([2*a, 2*a, 2*a, 2*a, 2*a, 2*a, 4*a, 4*a, 4*a]) #3*a #separation between boxes

    data = np.fromfile(f'test_data.dat', dtype=float, sep=' ')
    data = data.reshape((-1, 4))

    N2_mean, N2_std, N_stats = countoscope.Calc_and_Output_Stats(data=data,
                                                                window_size_x=Lx, window_size_y=Ly, 
                                                                box_sizes=Box_Ls, sep_sizes=sep)
    
    # now do what you want with the data!
    # I would save it into a numpy archive with np.savez:
    np.savez('test_data/counted.npy', N2_mean=N2_mean, N2_std=N2_std, N_stats=N_stats, box_sizes=Box_Ls, sep_sizes=sep)

    # but for backward compatibility or for further processing outside of Python, you could also do this:
    for i, L in enumerate(N_stats[:, 0]):
        np.savetxt(f'test_data/{L}_mean.txt', N2_mean[i, :], delimiter=' ', fmt='%.10f')
        np.savetxt(f'test_data/{L}_std.txt',  N2_std [i, :], delimiter=' ', fmt='%.10f')
    np.savetxt(f'test_data/N_stats.txt',  N_stats, delimiter=' ', fmt='%.10f')

                                                             
