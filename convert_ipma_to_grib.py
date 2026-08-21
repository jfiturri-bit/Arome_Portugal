#!/usr/bin/env python3
"""Convert IPMA AROME PT2 UVCOMP NetCDF to LuckGrib-friendly GRIB2.

Input is the IPMA file with a CF Mercator grid and variable:
  UV(component=2, time=48, y=611, x=505), units m/s.
The script resamples onto a regular latitude/longitude grid and writes
UGRD/VGRD at 10 m for every forecast hour.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from math import ceil, floor, pi, log, tan, degrees
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from eccodes import (
    codes_grib_new_from_samples,
    codes_set,
    codes_set_values,
    codes_write,
    codes_release,
)

R_EARTH = 6378137.0
STEP_DEG = 0.025


def mercator_xy_from_lonlat(lon_deg: np.ndarray, lat_deg: np.ndarray):
    x = R_EARTH * np.deg2rad(lon_deg)
    y = R_EARTH * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_deg) / 2.0))
    return x, y


def lonlat_from_mercator_xy(x: np.ndarray, y: np.ndarray):
    lon = np.rad2deg(x / R_EARTH)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2.0)
    return lon, lat


def read_input(path: Path):
    with h5py.File(path, "r") as f:
        uv = np.asarray(f["UV"], dtype=np.float32)
        times = np.asarray(f["time"], dtype=np.int32)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        time_units = f["time"].attrs["units"].decode() if isinstance(f["time"].attrs["units"], bytes) else str(f["time"].attrs["units"])
        wind_components = np.asarray(f["wind_component"], dtype=np.int32)
    if uv.ndim != 4 or uv.shape[0] != 2:
        raise ValueError(f"Unexpected UV shape: {uv.shape}")
    if wind_components.tolist() != [0, 1]:
        raise ValueError(f"Unexpected wind_component values: {wind_components}")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x coordinate is not strictly increasing")
    # IPMA file stores y north-to-south. Flip to ascending for interpolator.
    if np.any(np.diff(y) < 0):
        y_asc = y[::-1]
        uv = uv[:, :, ::-1, :]
    elif np.all(np.diff(y) > 0):
        y_asc = y
    else:
        raise ValueError("y coordinate is not monotonic")
    prefix = "hours since "
    if not time_units.lower().startswith(prefix):
        raise ValueError(f"Unsupported time units: {time_units}")
    ref = datetime.fromisoformat(time_units[len(prefix):].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    return uv, times, x, y_asc, ref


def build_target_grid(x: np.ndarray, y: np.ndarray):
    lon_src, _ = lonlat_from_mercator_xy(x, np.array([y[0], y[-1]]))
    _, lat_src = lonlat_from_mercator_xy(np.array([x[0], x[-1]]), y)
    lon_min, lon_max = float(x.min() / R_EARTH * 180.0 / pi), float(x.max() / R_EARTH * 180.0 / pi)
    lat_min = float(lat_src.min())
    lat_max = float(lat_src.max())
    nlon = int(floor((lon_max - lon_min) / STEP_DEG)) + 1
    nlat = int(floor((lat_max - lat_min) / STEP_DEG)) + 1
    lons = lon_min + np.arange(nlon, dtype=np.float64) * STEP_DEG
    lats = lat_min + np.arange(nlat, dtype=np.float64) * STEP_DEG
    lon2d, lat2d = np.meshgrid(lons, lats)
    tx, ty = mercator_xy_from_lonlat(lon2d, lat2d)
    return lats, lons, tx, ty


def sample_uv(uv_tyx: np.ndarray, x: np.ndarray, y: np.ndarray, tx: np.ndarray, ty: np.ndarray):
    points = np.column_stack([ty.ravel(), tx.ravel()])
    out = np.empty((2, tx.shape[0], tx.shape[1]), dtype=np.float32)
    for c in range(2):
        interp = RegularGridInterpolator(
            (y, x), uv_tyx[c], method="linear", bounds_error=False, fill_value=np.nan
        )
        vals = interp(points).reshape(tx.shape).astype(np.float32)
        # Tiny numerical extrapolation at the boundary can be missing; nearest fill.
        if np.isnan(vals).any():
            nearest = RegularGridInterpolator(
                (y, x), uv_tyx[c], method="nearest", bounds_error=False, fill_value=np.nan
            )
            nv = nearest(points).reshape(tx.shape).astype(np.float32)
            vals = np.where(np.isnan(vals), nv, vals)
        out[c] = vals
    return out


def set_grid_keys(gid, lats: np.ndarray, lons: np.ndarray):
    codes_set(gid, "gridType", "regular_ll")
    codes_set(gid, "Ni", int(lons.size))
    codes_set(gid, "Nj", int(lats.size))
    codes_set(gid, "latitudeOfFirstGridPointInDegrees", float(lats[0]))
    codes_set(gid, "longitudeOfFirstGridPointInDegrees", float(lons[0]))
    codes_set(gid, "latitudeOfLastGridPointInDegrees", float(lats[-1]))
    codes_set(gid, "longitudeOfLastGridPointInDegrees", float(lons[-1]))
    codes_set(gid, "iDirectionIncrementInDegrees", STEP_DEG)
    codes_set(gid, "jDirectionIncrementInDegrees", STEP_DEG)
    codes_set(gid, "jScansPositively", 1)


def write_field(gid, path, ref, step_hour, lats, lons, values, param_number, short_name):
    codes_set(gid, "edition", 2)
    codes_set(gid, "discipline", 0)
    # Meteorological parameter: Momentum, wind, category 2.
    codes_set(gid, "parameterCategory", 2)
    codes_set(gid, "parameterNumber", param_number)
    codes_set(gid, "typeOfFirstFixedSurface", 103)
    codes_set(gid, "scaleFactorOfFirstFixedSurface", 0)
    codes_set(gid, "scaledValueOfFirstFixedSurface", 10)
    codes_set(gid, "productDefinitionTemplateNumber", 0)
    codes_set(gid, "typeOfGeneratingProcess", 2)
    codes_set(gid, "centre", 86)
    codes_set(gid, "dataDate", ref[0] * 10000 + ref[1] * 100 + ref[2])
    codes_set(gid, "dataTime", ref[3] * 100 + ref[4])
    codes_set(gid, "forecastTime", int(step_hour))
    codes_set(gid, "stepUnits", 1)
    codes_set(gid, "bitsPerValue", 16)
    codes_set(gid, "packingType", "grid_simple")
    set_grid_keys(gid, lats, lons)
    arr = np.asarray(values, dtype=np.float64).ravel()

    if not np.all(np.isfinite(arr)):
      arr = arr.copy()
      missing = ~np.isfinite(arr)

      codes_set(gid, "bitmapPresent", 1)
      codes_set(gid, "missingValue", 9999.0)

      arr[missing] = 9999.0

    codes_set_values(gid, arr)
    codes_write(gid, path)
    codes_release(gid)


def make_grib(input_path: Path, output_path: Path):
    uv, times, x, y, ref = read_input(input_path)
    lats, lons, tx, ty = build_target_grid(x, y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out:
        for idx, hour in enumerate(times.tolist()):
            sampled = sample_uv(uv[:, idx], x, y, tx, ty)
            ref_tuple = (ref.year, ref.month, ref.day, ref.hour, ref.minute)
            for comp, param, short_name in ((0, 2, "10u"), (1, 3, "10v")):
                gid = codes_grib_new_from_samples("regular_ll_sfc_grib2")
                # shortName is set when possible; parameter keys above remain authoritative.
                try:
                    codes_set(gid, "shortName", short_name)
                except Exception:
                    pass
                write_field(gid, out, ref_tuple, int(hour), lats, lons, sampled[comp], param, short_name)
    print(f"Wrote {output_path} with {len(times) * 2} messages, {lats.size}x{lons.size} grid")
    print(f"Lon {lons[0]:.4f} .. {lons[-1]:.4f}; Lat {lats[0]:.4f} .. {lats[-1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()
    make_grib(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
