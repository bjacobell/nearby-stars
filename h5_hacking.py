import numpy as np 
import hdf5plugin
import h5py 

file = '/datax/scratch/benjb/nenufar/AI-ATLAS_D20260210T235610_chan143-143_lane0_beam0.h5'

outdir = '/datax/scratch/benjb/nenufar/'

file_path = file

with h5py.File(file_path, 'a') as f:
    # Step 1: Read existing dataset
    dset = f['data']
    print(dset)
    data_uint8 = dset[()]
    print(data_uint8.shape)

    # Save attributes (important for Breakthrough Listen files)
    attrs = dict(dset.attrs)

    # Optional: preserve chunking/compression if present
    chunks = dset.chunks
    compression = dset.compression
    compression_opts = dset.compression_opts

    # Step 2: Convert to float32
    data_float = data_uint8.astype(np.float32)

    # Step 3: Delete original dataset
    del f['data']

    # Step 4: Create new dataset with same name
    dset_new = f.create_dataset(
        'data',
        data=data_float,
        dtype='float32',
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts
    )

    # Step 5: Restore attributes
    for key, value in attrs.items():
        dset_new.attrs[key] = value

    dset_new.attrs['nfpc'] = int(data_uint8.shape[-1]//12)