import numpy as np
import Numba_Box_Count_Stats as countoscope

import matplotlib.pyplot as plt # for plotting purposes
from matplotlib import cm

if __name__ == '__main__':
    Lx = 100.0 # box size x-dir 
    Ly = 100.0 # box size y-dir
    Box_Ls = np.array([16.0, 10.0, 4.0, 2.0, 1.0]) # array of box sizes to probe

    a = 1.0 #radius of particles
    sep = np.array([2*a,2*a, 2*a, 2*a, 4*a]) #separation between boxes
    # conservative separation is to keep about a particle radius in between boxes
    # for smaller boxes if you don't want the code to run too long it's good to keep the separation large
    
    
    # load data
    filename = 'example_dataset'
    folder = 'test_data/'
    extension = '.txt'

    data = np.fromfile(f'{folder}{filename}{extension}', dtype=float, sep=' ') # load data array as a single big line
    data = data.reshape((-1, 3)) # reshape it so that it has x / y / t on each row
    print(data)

    # run the main counting code
    N2_mean, N2_std, N_stats = countoscope.Calc_and_Output_Stats(data=data,
                                                                window_size_x=Lx, window_size_y=Ly, 
                                                                box_sizes=Box_Ls, sep_sizes=sep)
    
    # now do what you want with the data!
    # I would save it into a numpy archive with np.savez:
    np.savez(f'{folder}{filename}_counted.npy', N2_mean=N2_mean, N2_std=N2_std, N_stats=N_stats, box_sizes=Box_Ls, sep_sizes=sep)

    # but for backward compatibility or for further processing outside of Python, you could also do this:
    for i, L in enumerate(N_stats[:, 0]):
        np.savetxt(f'{folder}{filename}_{L}_mean.txt', N2_mean[i, :], delimiter=' ', fmt='%.10f')
        np.savetxt(f'{folder}{filename}{L}_std.txt',  N2_std [i, :], delimiter=' ', fmt='%.10f')
    np.savetxt(f'{folder}{filename}N_stats.txt',  N_stats, delimiter=' ', fmt='%.10f')                                                     


    # a few plots for sanity checks 
    
    cmap = cm.viridis #pick a great colormap
    num_Boxes = len(Box_Ls) #number of boxes 
    dt = 1.0 #time step in between frames
    nt = len(N2_mean[0][:])

    # basic plot
    for i, L in enumerate(N_stats[:, 0]):
        plt.loglog([it*dt for it in range(nt)],N2_mean[i][:],color = cmap(i/num_Boxes),label =f'Box Size  = {L}')

    axes = plt.gca()
    axes.set_ylim(1e-5, 1e-1)
    axes.set_xlabel('Time t')
    axes.set_ylabel('Number fluctuations \n $< (N(t) - N(0))^2 >$ ')
    plt.legend()
    plt.show()

    # make also rescaled plot
    for i, L in enumerate(N_stats[:, 0]):
        plt.loglog([it*dt/L**2 for it in range(nt)],[n2/L**2 for n2 in N2_mean[i][:]],color = cmap(i/num_Boxes),label =f'Box Size  = {L}')

    axes = plt.gca()
    axes.set_ylim(1e-7, 1e-2)
    axes.set_xlabel('Rescaled Time t/L^2')
    axes.set_ylabel('Rescaled Number fluctuations \n $< (N(t) - N(0))^2 > / Var(N)$ ')
    plt.legend()
    plt.show()
