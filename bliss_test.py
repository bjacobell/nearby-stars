import numpy as np 
import os
import sys
sys.path.append("/home/benjb/bliss/build/bliss/python")
import blissdedrift as bliss

#for i in range(3):
#    j = i+1

file = '/datax/scratch/benjb/bliss_LSCX_test/spliced_blc4041424344454647_guppi_58885_63646_MESSIER081_0057.rawspec.0000.h5'
outdir = '/datax/scratch/benjb/bliss_LSCX_test/bliss_output/'

snr_list = np.concatenate((np.arange(start=5, stop=36, step=5), [40, 50, 75, 100, 150, 200]))
#snr_list = np.array([1405])
l1_list = np.array([2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,75,100,125,150,175])

for snr in snr_list:
    for l1 in l1_list:
        console = 'bliss_find_hits ' + file + f' -e /datax/scratch/benjb/bliss_LSCX_test/bliss_output/GBT_spliced_PFB_response.f32 -md -4 -MD 4 -s {snr} --distance {l1} --number-coarse 512 --nchan-per-coarse 1048576 --output ' + outdir + os.path.basename(file)[:-3] + f'_SNR_{snr}_L1_{l1}.hits'
        os.system(console)
        console = 'bliss_hits_to_dat -i ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits -o ' + outdir + os.path.basename(file)[:-3] + f'_SNR_{snr}_L1_{l1}.dat'
        os.system(console)

# for i in range(9):
#     j = i+1

#     file = f'/datax/scratch/benjb/bliss_voyager_test/injections/synthetic_data_2000_injections_{j}.0000.h5'
#     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/'

#     #snr_list = np.arange(start=5, stop=36)
#     snr_list = np.array([40, 50, 75, 100, 250, 500, 750, 1000, 1250])

#     for snr in snr_list:
#         console = 'bliss_find_hits ' + file + f' -md -4 -MD 4 -s {snr} -c 0 --number-coarse 2000 --nchan-per-coarse 7158 --output ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits'
#         os.system(console)
#         console = 'bliss_hits_to_dat -i ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits -o ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.dat'
#         os.system(console)

# for i in range(10):
#     j = i

#     file = f'/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/synthetic_data_2000_injections_{j}.0000.h5'
#     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/'

#     #snr_list = np.arange(start=5, stop=36)
#     snr_list = np.array([40, 50, 75, 100, 250, 500, 750, 1000, 1250])

#     for snr in snr_list:
#         console = 'bliss_find_hits ' + file + f' -md -4 -MD 4 -s {snr} -c 0 --number-coarse 2000 --nchan-per-coarse 7158 --output ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits'
#         os.system(console)
#         console = 'bliss_hits_to_dat -i ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.hits -o ' + outdir + os.path.basename(file)[:-3] + f'_{snr}.dat'
#         os.system(console)