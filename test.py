from astropy import units as u
import setigen as stg
import numpy as np
import matplotlib.pyplot as plt
#import h5py
import os
import blimpy as bl
from astropy.coordinates import Angle
output_file = '/datax/scratch/benjb/test.h5'
#os.makedirs('et_signals', exist_ok=True)


head = {'DIMENSION_LABELS': np.array([b'time', b'feed_id', b'frequency'], dtype=object), 
        'az_start': 0.0, 
        'data_type': 1, 
        'fch1': 6095.214842353016, # This is the edge of a C-band node; it can be changed, but other values below will then also have to be changed.
        'foff': -2.7939677238464355e-06, # Fine-channel frequency resolution.
        'machine_id': 20, 
        'nbits': 32, 
        'nchans': 1048576*16, # 16 coarse channels of 2e20 fine channels each.
        'nifs': 1, 
        'source_name': 'synthetic', # Placeholder name. Some Blimpy functions care about the source name.
        'src_dej': Angle(0*u.deg), # Placeholder declination.
        'src_raj': Angle('0h0m0s'), # Placeholder RA.
        'telescope_id': 6, 
        'tsamp': 18.253611008, # Time resolution.
        'tstart': 60000, # Arbitrary epoch of observation.
        'za_start': 0.0}

print('Creating frame ...')
frame = stg.Frame(
    fchans=1048576 * u.pixel,
    tchans=16 * u.pixel,
    df=2.7939677238464355 * u.Hz,
    dt=18.253611008 * u.s,
    fch1=8421.386717353016 * u.MHz,
    dim_order='fht'
)
frame.add_noise(x_mean=10, noise_type='chi2')
print('Adding signal ...')
frame.add_constant_signal(
    f_start=frame.get_frequency(200_000),
    drift_rate=np.random.uniform(-2, 2) * u.Hz / u.s,
    level=frame.get_intensity(snr=np.random.randint(10, 15)),
    width=np.random.randint(20, 40) * u.Hz,
    f_profile_type='sinc2'
)
print('Saving ...')
#frame.save_h5(output_file)
fb_new = bl.Waterfall(filename=None, header_dict=head, data_array=np.expand_dims(np.flip(frame.data, axis=1), axis=1).astype('<f4'))
print(fb_new.data.dtype)
fb_new.write_to_hdf5(output_file)

print(frame.data.dtype)
print('seticore ...')
console = 'seticore ' + output_file + ' -M 4 -s ' + str(10) + ' --output ' + '/datax/scratch/benjb/test_seticore.dat'
os.system(console)