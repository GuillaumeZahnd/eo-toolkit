import calendar
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import requests

from fetch_stac import NoSceneFoundError, fetch_s1, fetch_s2

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CHANNELS = ("t_mean", "precip", "et0", "water_balance")


def fetch_weather_monthly(lon: float, lat: float, periods: Sequence[str]) -> npt.NDArray[np.float32]:
    """Monthly weather aggregates at a point from the Open-Meteo ERA5 archive.

    Parameters
    ----------
    lon, lat : float
        Point in decimal degrees. Reanalysis cells are ~10-30 km, so this is
        effectively one value for the whole ROI.
    periods : Sequence[str]
        Months as ``'YYYY-MM'``.

    Returns
    -------
    numpy.ndarray
        Float32 array ``(len(periods), 4)`` with columns :data:`WEATHER_CHANNELS`:
        mean temperature (degC), total precipitation (mm), total reference
        evapotranspiration (mm) and ``precip - et0`` (mm, a drought-stress proxy).
        Months without data are NaN.
    """
    last_y, last_m = map(int, max(periods).split("-"))
    r = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{min(periods)}-01",
            "end_date": f"{max(periods)}-{calendar.monthrange(last_y, last_m)[1]:02d}",
            "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
            "timezone": "UTC",
        },
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    month = np.array([t[:7] for t in d["time"]])
    temp = np.array(d["temperature_2m_mean"], dtype=float)  # JSON nulls -> NaN
    precip = np.array(d["precipitation_sum"], dtype=float)
    et0 = np.array(d["et0_fao_evapotranspiration"], dtype=float)

    out = np.full((len(periods), len(WEATHER_CHANNELS)), np.nan, dtype=np.float32)
    for i, p in enumerate(periods):
        sel = month == p
        if not sel.any() or np.isnan(precip[sel]).all():
            continue
        p_sum, e_sum = np.nansum(precip[sel]), np.nansum(et0[sel])
        out[i] = (np.nanmean(temp[sel]), p_sum, e_sum, p_sum - e_sum)
    return out


def build_sample(
    lon: float,
    lat: float,
    size_m: float,
    years: Sequence[int],
    s2_bands: Sequence[str],
    s2_url: str,
    s1_url: str,
    *,
    months: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
    s1_bands: Sequence[str] = ("vv", "vh"),
    max_cloud: float = 30.0,
    orbit_state: str = "descending",
    relative_orbit: int | None = None,
    resolution: float = 10.0,
) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
    """Co-registered monthly time series of S2, S1 and weather around a point.

    Per month: the least cloudy Sentinel-2 scene, the temporal median of
    Sentinel-1 (fixed orbit, dB), and the monthly weather aggregates. S2 and S1
    use the same bbox, CRS and resolution, so they share a pixel grid (cropped
    to the smallest common shape if they differ by a pixel). Weather is
    broadcast as constant channels.

    Parameters
    ----------
    lon, lat : float
        Centre of the ROI in decimal degrees.
    size_m : float
        Side length of the square ROI, in metres.
    years : Sequence[int]
        Years to include. Use several: gradual decline shows up against a baseline.
    s2_bands : Sequence[str]
        Sentinel-2 asset keys, in output order.
    s2_url, s1_url : str
        STAC endpoints for S2 and S1.
    months : Sequence[int], optional
        Months to include each year. Default April-October (leaf-on season).
    s1_bands : Sequence[str], optional
        Sentinel-1 polarisations. Default ``("vv", "vh")``.
    max_cloud : float, optional
        Scene-level cloud cover limit for S2, in percent. Default ``30``.
    orbit_state : str, optional
        Sentinel-1 orbit direction. Default ``"descending"``.
    relative_orbit : int | None, optional
        Sentinel-1 track; set it for a consistent incidence angle.
    resolution : float, optional
        Pixel size in metres. Default ``10``.

    Returns
    -------
    x : numpy.ndarray
        Float32 array ``(T, C, H, W)``, ``T = len(years) * len(months)``.
        Channels: S2 bands, S1 bands (dB), then weather. Missing data is NaN.
    meta : dict
        ``periods`` (``'YYYY-MM'`` per step), ``channels``, ``s2_dates``,
        ``s1_dates`` (``None`` where missing) and ``valid`` (bool arrays ``s2``,
        ``s1``, one entry per time step).

    Raises
    ------
    NoSceneFoundError
        If neither S2 nor S1 returned anything for any month.
    """
    periods = [f"{y}-{m:02d}" for y in years for m in months]

    s2: list[tuple[npt.NDArray[np.float32], str] | None] = []
    s1: list[tuple[npt.NDArray[np.float32], str] | None] = []
    for p in periods:
        y, m = int(p[:4]), int(p[5:])
        start, end = f"{p}-01", f"{p}-{calendar.monthrange(y, m)[1]:02d}"
        try:
            s2.append(
                fetch_s2(
                    lon, lat, size_m, start, end, s2_bands, s2_url,
                    max_cloud=max_cloud, resolution=resolution,
                )
            )
        except NoSceneFoundError:
            s2.append(None)
        try:
            s1.append(
                fetch_s1(
                    lon, lat, size_m, start, end, s1_url,
                    bands=s1_bands, orbit_state=orbit_state, relative_orbit=relative_orbit,
                    reducer="median", to_db=True, resolution=resolution,
                )
            )
        except NoSceneFoundError:
            s1.append(None)

    found = [a for a, _ in (*(r for r in s2 if r), *(r for r in s1 if r))]
    if not found:
        raise NoSceneFoundError("No Sentinel-2 or Sentinel-1 scene found for any period.")
    h = min(a.shape[1] for a in found)
    w = min(a.shape[2] for a in found)

    weather = fetch_weather_monthly(lon, lat, periods)
    n2, n1, nw = len(s2_bands), len(s1_bands), len(WEATHER_CHANNELS)

    x = np.full((len(periods), n2 + n1 + nw, h, w), np.nan, dtype=np.float32)
    for t in range(len(periods)):
        if s2[t] is not None:
            x[t, :n2] = s2[t][0][:, :h, :w]
        if s1[t] is not None:
            x[t, n2 : n2 + n1] = s1[t][0][:, :h, :w]
        x[t, n2 + n1 :] = weather[t][:, None, None]

    meta: dict[str, Any] = {
        "periods": periods,
        "channels": [*s2_bands, *s1_bands, *WEATHER_CHANNELS],
        "s2_dates": [r[1] if r else None for r in s2],
        "s1_dates": [r[1] if r else None for r in s1],
        "valid": {
            "s2": np.array([r is not None for r in s2]),
            "s1": np.array([r is not None for r in s1]),
        },
    }
    return x, meta
