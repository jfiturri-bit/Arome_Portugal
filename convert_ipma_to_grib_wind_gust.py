#!/usr/bin/env python3
"""Convert IPMA AROME PT2 UVCOMP + GUSTMOD NetCDF to LuckGrib-friendly GRIB2.

Inputs:
  wind: UV(component=2, time=48, y=611, x=505)
  gust: UV(time=48, y=611, x=505)

Both IPMA datasets carry scale_factor metadata; it is applied before writing.
Output contains U10, V10 and 10 m wind gust for each forecast hour.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import floor, pi
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


def attr_num(ds, name, default):
    v = ds.attrs.get(name, default)
    return float(np.asarray(v).squeeze())


def decode_units(v):
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def apply_scale(ds):
    a = np.asarray(ds, dtype=np.float32)
    scale = attr_num(ds, "scale_factor", 1.0)
    offset = attr_num(ds, "add_offset", 0.0)
    return a * scale + offset


def lonlat_from_mercator_xy(x, y):
    lon = np.rad2deg(x / R_EARTH)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2.0)
    return lon, lat


def mercator_xy_from_lonlat(lon_deg, lat_deg):
    x = R_EARTH * np.deg2rad(lon_deg)
    y = R_EARTH * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_deg) / 2.0))
    return x, y


def parse_ref(time_ds):
    units = decode_units(time_ds.attrs["units"])
    prefix = "hours since "
    if not units.lower().startswith(prefix):
        raise ValueError(f"Unsupported time units: {units}")
    return datetime.fromisoformat(units[len(prefix):].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def read_wind(path):
    with h5py.File(path, "r") as f:
        ds = f["UV"]
        uv = apply_scale(ds)
        times = np.asarray(f["time"], dtype=np.int32)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        ref = parse_ref(f["time"])
        comps = np.asarray(f["wind_component"], dtype=np.int32)
    if uv.ndim != 4 or uv.shape[0] != 2:
        raise ValueError(f"Unexpected wind UV shape: {uv.shape}")
    if comps.tolist() != [0, 1]:
        raise ValueError(f"Unexpected wind_component values: {comps}")
    return uv, times, x, y, ref


def read_gust(path):
    with h5py.File(path, "r") as f:
        ds = f["UV"]
        gust = apply_scale(ds)
        times = np.asarray(f["time"], dtype=np.int32)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        ref = parse_ref(f["time"])
    if gust.ndim != 3:
        raise ValueError(f"Unexpected gust shape: {gust.shape}")
    return gust, times, x, y, ref


def make_y_ascending(data, y, y_axis):
    if np.all(np.diff(y) > 0):
        return data, y
    if np.all(np.diff(y) < 0):
        return np.flip(data, axis=y_axis), y[::-1]
    raise ValueError("y coordinate is not monotonic")


def build_target_grid(x, y):
    _, lat_src = lonlat_from_mercator_xy(np.array([x[0], x[-1]]), y)
    lon_min = float(x.min() / R_EARTH * 180.0 / pi)
    lon_max = float(x.max() / R_EARTH * 180.0 / pi)
    lat_min = float(lat_src.min())
    lat_max = float(lat_src.max())
    nlon = int(floor((lon_max - lon_min) / STEP_DEG)) + 1
    nlat = int(floor((lat_max - lat_min) / STEP_DEG)) + 1
    lons = lon_min + np.arange(nlon, dtype=np.float64) * STEP_DEG
    lats = lat_min + np.arange(nlat, dtype=np.float64) * STEP_DEG
    lon2d, lat2d = np.meshgrid(lons, lats)
    tx, ty = mercator_xy_from_lonlat(lon2d, lat2d)
    return lats, lons, tx, ty


def sample_scalar(field_yx, x, y, tx, ty):
    points = np.column_stack([ty.ravel(), tx.ravel()])
    interp = RegularGridInterpolator((y, x), field_yx, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(points).reshape(tx.shape).astype(np.float32)
    if np.isnan(vals).any():
        nearest = RegularGridInterpolator((y, x), field_yx, method="nearest",
                                          bounds_error=False, fill_value=np.nan)
        nv = nearest(points).reshape(tx.shape).astype(np.float32)
        vals = np.where(np.isnan(vals), nv, vals)
    return vals


def set_grid_keys(gid, lats, lons):
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


def write_field(out, ref, hour, lats, lons, values, param_number, short_name):
    gid = codes_grib_new_from_samples("regular_ll_sfc_grib2")
    codes_set(gid, "edition", 2)
    codes_set(gid, "discipline", 0)
    codes_set(gid, "parameterCategory", 2)
    codes_set(gid, "parameterNumber", int(param_number))
    codes_set(gid, "typeOfFirstFixedSurface", 103)
    codes_set(gid, "scaleFactorOfFirstFixedSurface", 0)
    codes_set(gid, "scaledValueOfFirstFixedSurface", 10)
    codes_set(gid, "productDefinitionTemplateNumber", 0)
    codes_set(gid, "typeOfGeneratingProcess", 2)
    codes_set(gid, "centre", 86)
    codes_set(gid, "dataDate", ref.year * 10000 + ref.month * 100 + ref.day)
    codes_set(gid, "dataTime", ref.hour * 100 + ref.minute)
    codes_set(gid, "forecastTime", int(hour))
    codes_set(gid, "stepUnits", 1)
    codes_set(gid, "bitsPerValue", 16)
    codes_set(gid, "packingType", "grid_simple")
    set_grid_keys(gid, lats, lons)
    try:
        codes_set(gid, "shortName", short_name)
    except Exception:
        pass

    arr = np.asarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)):
        arr = arr.copy()
        missing = ~np.isfinite(arr)
        codes_set(gid, "bitmapPresent", 1)
        codes_set(gid, "missingValue", 9999.0)
        arr[missing] = 9999.0

    codes_set_values(gid, arr)
    codes_write(gid, out)
    codes_release(gid)


def make_grib(wind_path, gust_path, output_path):
    uv, wt, xw, yw, rw = read_wind(wind_path)
    gust, gt, xg, yg, rg = read_gust(gust_path)

    if rw != rg:
        raise ValueError(f"Wind/gust run mismatch: {rw} vs {rg}")
    if not np.array_equal(wt, gt):
        raise ValueError("Wind/gust forecast times differ")
    if not np.allclose(xw, xg) or not np.allclose(yw, yg):
        raise ValueError("Wind/gust grids differ")

    uv, y = make_y_ascending(uv, yw, 2)
    gust, yg2 = make_y_ascending(gust, yg, 1)
    if not np.allclose(y, yg2):
        raise ValueError("Wind/gust y grids differ after orientation")

    lats, lons, tx, ty = build_target_grid(xw, y)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as out:
        for idx, hour in enumerate(wt.tolist()):
            u = sample_scalar(uv[0, idx], xw, y, tx, ty)
            v = sample_scalar(uv[1, idx], xw, y, tx, ty)
            g = sample_scalar(gust[idx], xw, y, tx, ty)
            write_field(out, rw, hour, lats, lons, u, 2, "10u")
            write_field(out, rw, hour, lats, lons, v, 3, "10v")
            # WMO GRIB2 discipline 0, category 2, parameter 22 = wind speed (gust)
            write_field(out, rw, hour, lats, lons, g, 22, "gust")

    print(f"Wrote {output_path}: {len(wt) * 3} messages")
    print(f"Run {rw.isoformat()} | Lon {lons[0]:.4f}..{lons[-1]:.4f} | Lat {lats[0]:.4f}..{lats[-1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wind")
    ap.add_argument("gust")
    ap.add_argument("output")
    args = ap.parse_args()
    make_grib(Path(args.wind), Path(args.gust), Path(args.output))


if __name__ == "__main__":
    main()
