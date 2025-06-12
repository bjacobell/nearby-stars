import numpy as np 
import matplotlib.pyplot as plt 
import glob
import os
import pandas as pd
from turbo_seti.find_doppler.find_doppler import FindDoppler


file = '/datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_low_drift_every_2_kHz.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_test.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_many_widths_many_drifts.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/Voyager_injections_unresolved_evenmorespace.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/ANDXIV_S_injections_unresolved_evenmorespace.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/synthetic_unresolved_vardrift1_32kHz.h5'
file = '/datax/scratch/benjb/bliss_LSCX_test/synthetic_unresolved_vardrift1_32kHz_snr1_1000.h5'
file = '/datag/pipeline/AGBT23A_999_43/blc44_blp04/blc44_guppi_60233_38409_GJ144_0016.rawspec.0000.h5'
outdir = '/datax/scratch/benjb/bliss_LSCX_test/'

#console = 'seticore ' + file + ' -M 4 -s 10 --output ' + outdir + os.path.basename(file)[:-2] + 'dat'
#os.system(console)

snr_list = np.concatenate((np.arange(start=5, stop=36), [40, 50, 75, 100, 125, 150, 200]))
snr_list = np.array([2,3,4])
snr_list = np.array([6])
#snr_list = [10]

import time

start = time.time()

for snr in snr_list:
    print('SNR ' + str(snr))
    # doppler = FindDoppler(file,
    #               min_drift = 1e-5,
    #               max_drift = 4,
    #               snr = snr,       
    #               #out_dir = outdir + 'seticore/',
    #               out_dir = outdir,
    #               n_coarse_chan = 16
    #              )
    # doppler.search()
    console = 'seticore ' + file + ' -M 4 -s ' + str(snr) + ' --output ' + outdir + os.path.basename(file)[:-2] + str(snr) + '_seticore.dat'
    os.system(console)
    end = time.time()
    print(end-start)

# for i in range(10):
#     j = i

#     file = '/datax/scratch/benjb/bliss_voyager_test/injections/synthetic_data_2000_injections_'+str(j)+'.0000.h5'
#     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/'

#     snr_list = np.arange(start=5, stop=36)

#     for snr in snr_list:
#         # doppler = FindDoppler(file,
#         #               max_drift = 4,
#         #               snr = snr,       
#         #               out_dir = outdir + 'seticore/',
#         #               n_coarse_chan = 2000
#         #              )
#         # doppler.search()
#         console = 'seticore ' + file + ' -M 4 -s ' + str(snr) + ' -n 2000 --output ' + outdir + os.path.basename(file)[:-2] + 'seticore.dat'
#         os.system(console)

# for i in range(3):
#     j = i

#     file = '/datax/scratch/benjb/bliss_voyager_test/injections/synthetic_data_2000_injections_'+str(j)+'.0000.h5'
#     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_no_drift_injections/'

#     snr_list = np.arange(start=5, stop=36)

#     for snr in snr_list:
#         # doppler = FindDoppler(file,
#         #               max_drift = 4,
#         #               snr = snr,       
#         #               out_dir = outdir + 'seticore/',
#         #               n_coarse_chan = 2000
#         #              )
#         # doppler.search()
#         console = 'seticore ' + file + ' -M 4 -s ' + str(snr) + ' -n 2000 --output ' + outdir + os.path.basename(file)[:-2] + 'seticore.dat'
#         os.system(console)

# ##########

# snr_list = [6, 7]

# for snr in snr_list:

#     for i in range(3):

#         j = i+1

#         file = '/datax/scratch/benjb/bliss_voyager_test/injections/synthetic_data_2000_injections_'+str(j)+'.0000.h5'
#         outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/'

#         if not os.path.exists(outdir + 'seticore/' + str(snr)):
#             os.makedirs(outdir + 'seticore/' + str(snr))

#         doppler = FindDoppler(file,
#                       max_drift = 4,
#                       snr = snr,       
#                       out_dir = outdir + 'seticore/' + str(snr),
#                       n_coarse_chan = 2000,
#                       gpu_backend = True,
#                       blank_dc = True
#                      )
#         doppler.search()

    # for i in range(3):

    #     j = i

    #     file = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_no_drift_injections/synthetic_data_2000_injections_'+str(j)+'.0000.h5'
    #     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_no_drift_injections/'

    #     if not os.path.exists(outdir + 'seticore/' + str(snr)):
    #         os.makedirs(outdir + 'seticore/' + str(snr))

    #     doppler = FindDoppler(file,
    #                   max_drift = 4,
    #                   snr = snr,       
    #                   out_dir = outdir + 'seticore/' + str(snr),
    #                   n_coarse_chan = 2000,
    #                   gpu_backend = True,
    #                   blank_dc = True
    #                  )
    #     doppler.search()

    # for i in range(3):

    #     j = i

    #     file = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/synthetic_data_2000_injections_'+str(j)+'.0000.h5'
    #     outdir = '/datax/scratch/benjb/bliss_voyager_test/injections/high_SNR_injections/'

    #     if not os.path.exists(outdir + 'seticore/' + str(snr)):
    #         os.makedirs(outdir + 'seticore/' + str(snr))

    #     doppler = FindDoppler(file,
    #                   max_drift = 4,
    #                   snr = snr,       
    #                   out_dir = outdir + 'seticore/' + str(snr),
    #                   n_coarse_chan = 2000,
    #                   gpu_backend = True,
    #                   blank_dc = True
    #                  )
    #     doppler.search()

    