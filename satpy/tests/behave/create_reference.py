from dask.diagnostics import ProgressBar
from satpy import Scene
from glob import glob
import os
import warnings
import dask

os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
os.environ['PYTROLL_CHUNK_SIZE'] = '1024'
warnings.simplefilter('ignore')
dask.config.set(scheduler='threads', num_workers=4)

# Get the list of satellite files to open
satellite = "GOES16"
filenames = glob(f'./satellite_data/{satellite}/*.nc')

scn = Scene(reader='abi_l1b', filenames=filenames)

#  what composites Satpy knows how to make and that it has the inputs for?
print(scn.available_composite_names())

composite = 'airmass'
scn.load([composite])
with ProgressBar():
    scn.save_datasets(writer='simple_image', filename=f'./features/data/reference/reference_image_{satellite}_{composite}.png')



