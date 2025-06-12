import os

outdir = '/datax/scratch/benjb/bliss_LSCX_test/'
file = '/datag/pipeline/AGBT16A_999_219/holding/spliced_blc0001020304050607_guppi_57550_40640_GJ1002_0003.gpuspec.0000.h5'
file = '/datag/pipeline/AGBT16B_999_20/holding/spliced_blc0001020304050607_guppi_57635_34975_Gj144_0003.gpuspec.0000.h5'
console = f'bliss_find_hits {file} -e /datax/scratch/benjb/bliss_LSCX_test/bliss_output/GBT_spliced_PFB_response.f32 -d cuda:0 -md -4 -MD 4 -s 20 --number-coarse 512 --distance 30 --output ' + outdir + os.path.basename(file)[:-3] + f'_test.dat'
os.system(console)