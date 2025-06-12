import numpy as np 
import os
import sys
#import blimpy as bl
#sys.path.append("/home/benjb/bliss/build/bliss/python")
#import blissdedrift as bliss

#for i in range(3):
#    j = i+1

file = '/datax/scratch/benjb/bliss_LSCX_test/spliced_blc4041424344454647_guppi_58885_63646_MESSIER081_0057.rawspec.0000.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_low_drift_every_2_kHz.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_test.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_many_widths_many_drifts.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_unresolved_evenmorespace.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/ANDXIV_S_injections_unresolved_evenmorespace.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/synthetic_unresolved_vardrift1_32kHz.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/synthetic_unresolved_vardrift1_32kHz_snr1_1000.h5'
file = '/datag/pipeline/AGBT23A_999_43/blc44_blp04/blc44_guppi_60233_38409_GJ144_0016.rawspec.0000.h5'
#file = '/datag/pipeline/AGBT18B_999_07/blc43_blp03/blc03_guppi_58348_31667_HIP3301_0014.gpuspec.0000.h5'
#file = '/datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.h5'
outdir = '/datax/scratch/benjb/bliss_LSCX_test/'

snr_list = np.concatenate((np.arange(start=10, stop=36, step=5), [40, 50, 75, 100, 150, 200]))
snr_list = np.array([21])
#snr_list = np.array([30])
#snr_list = np.array([30])
#l1_list = np.array([26,27,28,29,31,32,33,34])
l1_list = np.array([30])
#l1_list = np.array([10])

#fb = bl.Waterfall(file)
#fb.info()
tvec = []

import time

start = time.time()

for snr in snr_list:
    for l1 in l1_list:
        console = f'bliss_find_hits {file} -e /datax/scratch/benjb/bliss_LSCX_test/bliss_output/GBT_spliced_PFB_response.f32 -d cuda:0 -md -4 -MD 4 -s {snr} --number-coarse 64 --distance {l1} --output ' + outdir + os.path.basename(file)[:-3] + f'_nosig_nosk_SNR_{snr}_L1_{l1}.dat'
        os.system(console)
        end = time.time()
        #print(end - start)
        tvec.append(end - start)
#         console = 'bliss_hits_to_dat -i ' + outdir + os.path.basename(file)[:-3] + f'_SNR_{snr}_L1_{l1}.hits -o ' + outdir + os.path.basename(file)[:-3] + f'_SNR_{snr}_L1_{l1}.dat'
#         os.system(console)
print(tvec)

#'bliss_find_hits /datag/pipeline/AGBT18B_999_07/blc43_blp03/blc03_guppi_58348_31667_HIP3301_0014.gpuspec.0000.h5 -e /datax/scratch/benjb/bliss_LSCX_test/0_0_GBT_channelizer_response.f32 -d cuda:0 -md -4 -MD 4 -s 30 --number-coarse 64 --filter-sigmaclip --filter-low-sk --filter-high-sk --distance 10 --output 000_test.dat'

# snr = 30
# l1 = 10
# console = f'bliss_find_hits /datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.h5 -e /datax/scratch/benjb/bliss_LSCX_test/bliss_output/GBT_spliced_PFB_response.f32 -d cuda:2 -md -4 -MD 4 -s 30 --number-coarse 1 --nchan-per-coarse 1048576 --filter-sigmaclip --filter-low-sk --filter-high-sk --distance 10 --output /datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.hits'
# os.system(console)
# console = 'bliss_hits_to_dat -i /datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.hits -o /datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.dat'
# os.system(console)

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