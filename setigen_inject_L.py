import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import blimpy as bl 
import setigen as stg 
from astropy import units as u
from astropy.coordinates import Angle
import time

#n_inj = 1000

#h5_path = '/datag/pipeline/AGBT20A_999_03/collate/spliced_blc4041424344454647_guppi_58885_63646_MESSIER081_0057.rawspec.0000.h5'
h5_path = '/datag/pipeline/AGBT19B_999_121/blc73_blp03/blc73_guppi_58832_16209_MESSIER031_0057.gpuspec.0000.h5'

# center_frequencies = np.concatenate(
#     (np.arange(1100+0.02, 1200, 0.02), np.arange(1340+0.02, 1900, 0.02))
# )

fb = bl.Waterfall(h5_path, load_data=False) # load_data=False means load only the header
head = fb.header
freq_range = (head['fch1'], head['fch1']+head['nchans']*head['foff'])
print(freq_range) # can't guarantee whether fch1 is a min or a max
min_freq = np.min(freq_range)
max_freq = np.max(freq_range)

center_frequencies = np.arange(min_freq+0.02, max_freq-0.02, 0.02)

n_inj = len(center_frequencies)
print(n_inj, 'hits to be added')

cf_offsets = np.random.uniform(-0.005, 0.005, n_inj)

center_frequencies = center_frequencies + cf_offsets

fdiffs = np.diff(np.sort(center_frequencies))
#center_frequencies = np.random.uniform(1100, 1900, n_inj)
#drift_rates = np.random.uniform(-4, 4, n_inj)
drift_rates = np.random.uniform(-2, 2, n_inj)
snrs = 1/np.random.power(3/2, n_inj)
def power_law(a, b, alpha, size=1):
    # power law from uniform distribution: https://stackoverflow.com/questions/31114330/python-generating-random-numbers-from-a-power-law-distribution
    uniform = np.random.random(size=size)
    aa, bb = a**alpha, b**alpha
    return (aa + (bb - aa)*uniform)**(1/alpha)
snrs = power_law(10, 1000, -1.5, size=n_inj)
snrs = np.random.uniform(10, 1000, n_inj)
#snrs = np.arange(1000) + 1
widths = np.random.randint(1, 11, n_inj)

#plt.hist(center_frequencies)
#plt.show()
print(np.min(fdiffs)*1000000)
print(len(center_frequencies))

wf = []
start = time.time()

print('Since waterfall appending began ...')
for j in range(n_inj):
    if j%100 == 0:
        print(f'  {j}: {time.time() - start} s elapsed')
    #wf.append(bl.Waterfall(h5_path, f_start=center_frequencies[j]-0.007, f_stop=center_frequencies[j]+0.007))
    wf.append(bl.Waterfall(h5_path, f_start=center_frequencies[j]-0.0035, f_stop=center_frequencies[j]+0.0035))

fb = bl.Waterfall(h5_path)
freqs, data = fb.grab_data()
header_dict = fb.header

for j in range(n_inj):
    c = stg.Frame(wf[j])
    block_freqs, _ = wf[j].grab_data()
    if j % 100 == 0:
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

fb_new.write_to_hdf5('/datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_blc73_guppi_58832_16209_MESSIER031_0057.rawspec.0000.h5')
np.save('/datax/scratch/benjb/bliss_LSCX_test/quick_test/widths_to_10_injections_freq_DR_snr_widths_MESSIER031_0057.npy', np.array([center_frequencies, drift_rates, snrs, widths]))
