# requires pytest-datafiles
import xarray as xr
import pandas as pd
import pytest
import numpy as np
import os
from easy_coloc import lib_easy_coloc

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files/',
    )


@pytest.mark.parametrize("out", ['ndarray', 'xarray'])
def test_ar07w(datafiles, out):
    ar07w = pd.read_csv(FIXTURE_DIR + 'AR07W_stations.txt',
                        skipinitialspace=True)
    ds = xr.open_dataset(FIXTURE_DIR + 'woa_labrador.nc',
                         decode_times=False)#,
                         #chunks={'time': 1, 'depth': 1})

    proj = lib_easy_coloc.projection(ar07w['lon'].values,
                                     ar07w['lat'].values,
                                     grid=ds)
    fld = proj.run(ds['t_an'], outtype=out)
    if out == 'ndarray':
        assert isinstance(fld, np.ndarray)
        assert fld.shape == (1, 1, 102, 30)
    elif out == 'xarray':
        assert isinstance(fld, xr.core.dataarray.DataArray)
        assert fld.shape == (1, 1, 102, 30)
        assert isinstance(fld.values, np.ndarray)

    return None
