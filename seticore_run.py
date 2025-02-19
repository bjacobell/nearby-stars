import numpy as np 
import matplotlib.pyplot as plt 
import glob
import os
import pandas as pd
from turbo_seti.find_doppler.find_doppler import FindDoppler


file = '/datax/scratch/benjb/bliss_LSCX_test/spliced_blc4041424344454647_guppi_58885_63646_MESSIER081_0057.rawspec.0000.h5'
outdir = '/datax/scratch/benjb/bliss_LSCX_test/seticore_output/L-band/'

#console = 'seticore ' + file + ' -M 4 -s 10 --output ' + outdir + os.path.basename(file)[:-2] + 'dat'
#os.system(console)

snr_list = np.concatenate((np.arange(start=5, stop=36), [40, 50, 75, 100, 125, 150, 200]))

for snr in snr_list:
    print('SNR ' + str(snr))
    # doppler = FindDoppler(file,
    #               max_drift = 4,
    #               snr = snr,       
    #               out_dir = outdir + 'seticore/',
    #               n_coarse_chan = 2000
    #              )
    # doppler.search()
    console = 'seticore ' + file + ' -M 4 -s ' + str(snr) + ' --output ' + outdir + os.path.basename(file)[:-2] + str(snr) + '_seticore.dat'
    os.system(console)

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

    