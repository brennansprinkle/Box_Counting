import numpy as np
import countoscope

import matplotlib.pyplot as plt # for plotting purposes
from matplotlib import cm

if __name__ == '__main__':
    Lx = 100.0 # box size x-dir 
    Ly = 100.0 # box size y-dir
    Box_Ls = np.array([16.0, 8.0, 4.0, 2.0, 1.0]) # array of box sizes to probe

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
    N2_mean, N2_std, N_stats = countoscope.calculate_nmsd(data=data,
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
        plt.loglog([it*dt for it in range(1, nt)], N2_mean[i, 1:], color=cmap(i/num_Boxes), label=f'Box Size = {L}')
        # the point at t=0 messes up a loglog graph so we don't plot it

    axes = plt.gca()
    axes.set_xlabel('Time $t$')
    axes.set_ylabel(r'Number fluctuations $\langle (N(t) - N(0))^2 \rangle$')
    plt.legend()
    plt.savefig('number_fluctuations.png')
    plt.show()
    plt.clf()

    # make also rescaled plot
    for i, L in enumerate(N_stats[:, 0]):
        plt.loglog([it*dt/L**2 for it in range(1, nt)], [n2/L**2 for n2 in N2_mean[i, 1:]], color=cmap(i/num_Boxes), label=f'Box Size = {L}')
        # the point at t=0 messes up a loglog graph so we don't plot it

    axes = plt.gca()
    axes.set_xlabel('Rescaled time $t/L^2$')
    axes.set_ylabel(r'Rescaled Number fluctuations $\langle (N(t) - N(0))^2 \rangle / Var(N)$')
    plt.legend()
    plt.savefig('number_fluctuations_rescaled.png')
    plt.show()