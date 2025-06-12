import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import blimpy as bl 
import setigen as stg 
from astropy import units as u
from astropy.coordinates import Angle
import time

#n_inj = 1000

h5_path = '/datag/pipeline/AGBT19B_999_118/collate0/spliced_blc0001020304050607_guppi_58818_09554_MESSIER081_0029.gpuspec.0000.h5'

center_frequencies = np.concatenate(
    (np.arange(1800+0.02, 2300, 0.02), np.arange(2360+0.02, 2700, 0.02))
)

n_inj = len(center_frequencies)
print(n_inj)

cf_offsets = np.random.uniform(-0.005, 0.005, n_inj)

center_frequencies = center_frequencies + cf_offsets

fdiffs = np.diff(np.sort(center_frequencies))
#center_frequencies = np.random.uniform(1100, 1900, n_inj)
drift_rates = np.random.uniform(-4, 4, n_inj)
snrs = 1/np.random.power(3/2, n_inj)
#snrs = np.arange(1000) + 1
widths = np.random.randint(1, 4, n_inj)

#plt.hist(center_frequencies)
#plt.show()
print(np.min(fdiffs)*1000000)
print(len(center_frequencies))

wf = []
start = time.time()

print('Since waterfall appending began ...')
for j in range(n_inj):
    if j%1000 == 0:
        print(f'  {j}: {time.time() - start} s elapsed')
    wf.append(bl.Waterfall(h5_path, f_start=center_frequencies[j]-0.007, f_stop=center_frequencies[j]+0.007))

fb = bl.Waterfall(h5_path)
freqs, data = fb.grab_data()
header_dict = fb.header

for j in range(n_inj):
    c = stg.Frame(wf[j])
    block_freqs, _ = wf[j].grab_data()
    if j % 1000 == 0:
        print(f'Adding signal {j} ...')
    c.add_signal(stg.constant_path(f_start=(center_frequencies[j])*u.MHz,
                               drift_rate=drift_rates[j]*u.Hz/u.s),
                           stg.constant_t_profile(level=c.get_intensity(snr=snrs[j])),
                           stg.sinc2_f_profile(width=widths[j]*c.df*u.Hz),
                           stg.constant_bp_profile(level=1),
                           doppler_smearing=True,
                           smearing_subsamples=15)
    data[:,np.where((freqs >= block_freqs[-1]) & (freqs <= block_freqs[0]))[0]] = np.flip(c.data, axis=1)

fb_new = bl.Waterfall(filename=None, header_dict=header_dict, data_array=np.expand_dims(data, axis=1))

fb_new.write_to_hdf5('/datax/scratch/benjb/bliss_LSCX_test/power_law/S/spliced_blc0001020304050607_guppi_58818_09554_MESSIER081_0029.gpuspec.0000.h5')
np.save('/datax/scratch/benjb/bliss_LSCX_test/power_law/S/injections_freq_DR_snr_widths_MESSIER081_0029.npy', np.array([center_frequencies, drift_rates, snrs, widths]))
