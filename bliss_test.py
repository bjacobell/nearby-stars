import numpy as np 
import os
import sys
sys.path.append("/home/benjb/bliss/build/bliss/python")
import blissdedrift as bliss

for i in range(10):
    j = i

    file = f'/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/synthetic_data_2000_injections_{j}.0000.h5'
    outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/'

    snr_list = np.arange(start=5, stop=36)

    for snr in snr_list:
        console = 'bliss_find_hits ' + file + f' -md -4 -MD 4 -s {snr} -c 0 --number-coarse 2000 --nchan-per-coarse 7158 --output ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits'
        os.system(console)
        console = 'bliss_hits_to_dat -i ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits -o ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.dat'
        os.system(console)