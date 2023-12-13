import time
import json
from typing import Dict
import torch
import numpy as np
from datetime import datetime, timedelta
import shutil


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig

# from lib.models.mlp import MLPConfig
from lib.serialization import serialize_human
from lib.train_dataclasses import TrainEpochState
from lib.metric import MetricSample
from lib.compute_env import env

from dataclasses import dataclass
import healpix
import xarray as xr
import experiments.weather.cdsmontly as cdstest


def numpy_to_xds(np_array, xds_template):
    transformed_ds = xr.Dataset()
    for i, var_name in enumerate(xds_template.data_vars):
        transformed_ds[var_name] = xr.DataArray(
            np_array[i], dims=xds_template.dims, coords=xds_template.coords
        )
    return transformed_ds


@dataclass
class DataHPConfig:
    nside: int = 64
    version: int = 10

    def serialize_human(self):
        return serialize_human(self.__dict__)


@dataclass
class DataSpecHP:
    nside: int
    n_surface: int
    n_upper: int


def days_between_years(start_year: int, end_year: int) -> int:
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    return (end_date - start_date).days


def day_index_to_era5_config(
    day_index: int, start_year: int, end_year: int
) -> cdstest.ERA5SampleConfig:
    start_date = datetime(start_year, 1, 1)
    target_date = start_date + timedelta(days=day_index)
    return cdstest.ERA5SampleConfig(
        year=target_date.strftime("%Y"),
        month=target_date.strftime("%m"),
        day=target_date.strftime("%d"),
        time="00:00:00",
    )


class DataHP(torch.utils.data.Dataset):
    def __init__(self, data_config: DataHPConfig):
        self.config = data_config

    @staticmethod
    def data_spec(config: DataHPConfig):
        return DataSpecHP(
            nside=config.nside,
            n_surface=4,
            n_upper=5
            # input_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            # output_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
            # target_shape=torch.Size([4, healpix.nside2npix(config.nside)]),
        )

    def e5_to_numpy(self, e5xr):
        npix = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(
            self.config.nside, np.arange(0, npix, 1), lonlat=True, nest=True
        )
        hlong = np.mod(hlong, 360)
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")

        def interpolate(variable: xr.DataArray):
            xhp = variable.interp(
                latitude=xlat, longitude=xlong, kwargs={"fill_value": None}
            )
            hp_image = np.array(xhp.to_array().to_numpy(), dtype=np.float32)
            hp_image = (hp_image - hp_image.mean(axis=1, keepdims=True)) / hp_image.std(
                axis=1, keepdims=True
            )
            return hp_image

        hp_surface = interpolate(e5xr.surface)
        hp_upper = interpolate(e5xr.upper)
        # max_vals = np.amax(hp_image, axis=1, keepdims=True)
        # hp_image = hp_image / max_vals
        # breakpoint()
        return hp_surface, hp_upper

    def get_driscoll_healy(self, idx):
        e5sc = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        e5_target_config = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="03:00:00"
        )
        e5target = cdstest.get_era5_sample(e5_target_config)
        return e5s, e5target

    def get_template_e5s(self):
        e5sc = cdstest.ERA5SampleConfig(
            year="1999", month="01", day="01", time="00:00:00"
        )
        e5s = cdstest.get_era5_sample(e5sc)
        return e5s

    def __getitem__(self, idx):
        fs_cache_path = env().paths.datasets / "era5_lite_np_cache" / f"{idx}"
        fs_cache_path_tmp = (
            env().paths.datasets / "era5_lite_np_cache" / f"{idx}_constructing"
        )
        names = dict(
            input_surface="surface.npy",
            input_upper="upper.npy",
            target_surface="target_surface.npy",
            target_upper="target_upper.npy",
        )
        item_dict = dict(
            sample_id=idx,
        )
        if fs_cache_path.is_dir():
            for key, filename in names.items():
                item_dict[key] = np.load(fs_cache_path / filename)
        else:
            e5s_input_config = day_index_to_era5_config(idx, 2007, 2017)
            e5s_target_config = cdstest.add_timedelta(e5s_input_config, days=1)
            e5s = cdstest.get_era5_sample(e5s_input_config)
            e5target = cdstest.get_era5_sample(e5s_target_config)
            hp_surface, hp_upper = self.e5_to_numpy(e5s)
            hp_target_surface, hp_target_upper = self.e5_to_numpy(e5target)
            data_dict = dict(
                input_surface=hp_surface,
                input_upper=hp_upper,
                target_surface=hp_target_surface,
                target_upper=hp_target_upper,
            )
            fs_cache_path_tmp.mkdir(parents=True, exist_ok=True)
            for key, filename in names.items():
                np.save(fs_cache_path_tmp / filename, data_dict[key])

            open(fs_cache_path_tmp / "era5config_input.json", "w").write(
                json.dumps(e5s_input_config.__dict__, indent=2)
            )
            open(fs_cache_path_tmp / "era5config_target.json", "w").write(
                json.dumps(e5s_target_config.__dict__, indent=2)
            )

            if not fs_cache_path.is_dir():
                shutil.move(fs_cache_path_tmp, fs_cache_path)

            item_dict.update(data_dict)

        return item_dict

    def __len__(self):
        return days_between_years(2007, 2017)
