from stac import search_stac, PLANETARY_COMPUTER_URL, bbox_around, NoSceneFoundError


def get_dominant_sentinel1_relative_orbit(
    lon: float,
    lat: float,
    size_m: float,
    start: str,
    end: str,
    orbit_state: str = "descending",
    stac_url: str = PLANETARY_COMPUTER_URL,
) -> tuple[int, str]:
    """Finds the dominant relative orbit and reports the best available orbit quality source.

    Tries quality tiers in order: POEORB -> RESORB -> Any available.

    Parameters
    ----------
    lon : float
        Longitude of the area's center point (WGS84).
    lat : float
        Latitude of the area's center point (WGS84).
    size_m : float
        Spatial window size in meters around the center point.
    start : str
        Start date in ISO format ("YYYY-MM-DD").
    end : str
        End date in ISO format ("YYYY-MM-DD").
    orbit_state : str, optional
        Orbit direction, either "descending" or "ascending" (default is "descending").
    stac_url : str, optional
        The STAC API endpoint URL.

    Returns
    -------
    tuple[int, str]
        A tuple containing (relative_orbit_number, orbit_source_tier).
    """
    bbox = bbox_around(lon, lat, size_m)
    base_query = {"sat:orbit_state": {"eq": orbit_state}}

    # Define quality tiers to attempt in descending order
    tiers = ["POEORB", "RESORB", None]

    for tier in tiers:
        q = dict(base_query)
        if tier is not None:
            q["s1:orbit_source"] = {"eq": tier}

        try:
            items = search_stac(stac_url, "sentinel-1-rtc", bbox, start, end, q)

            orbits = [
                i.properties.get("sat:relative_orbit")
                for i in items
                if i.properties.get("sat:relative_orbit") is not None
            ]

            if orbits:
                dominant_orbit = max(set(orbits), key=orbits.count)
                source_tier = tier if tier is not None else "UNKNOWN/DEFAULT"
                return dominant_orbit, source_tier

        except NoSceneFoundError:
            continue

    raise ValueError("No Sentinel-1 items found for the given bounding box and time range across any orbit tier.")
