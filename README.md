# AROME IPMA PT2 -> LuckGrib

This repository downloads the latest IPMA AROME PT2 10 m wind NetCDF, converts U/V wind components to a regular lat/lon GRIB2, and publishes one stable release asset for LuckGrib.

## One-time setup

1. Create a public GitHub repository.
2. Upload all files from this folder, keeping `.github/workflows/update.yml` in place.
3. Open Actions and run **Update AROME PT2 GRIB2** once with **Run workflow**.
4. After the run finishes, open Releases. The asset is:
   `arome-ipma-pt2-wind.grib2`
5. The stable download URL is:
   `https://github.com/YOUR_USER/YOUR_REPO/releases/download/latest/arome-ipma-pt2-wind.grib2`

The workflow also runs twice per day, offset from the top of the hour. IPMA publishes the mainland AROME wind product at 00 and 12 UTC, with 48 forecast hours.

## iPad shortcut

Create a Shortcut with these actions:

1. Text: the stable GitHub release URL above.
2. Get Contents of URL.
3. Save File to iCloud Drive, e.g. `AROME/`.
4. Open/Share the saved file with LuckGrib.

The first implementation intentionally automates only the 10 m vector wind (U/V). Gusts can be added as a second GRIB asset once the wind pipeline is stable.
