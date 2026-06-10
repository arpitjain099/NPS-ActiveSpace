import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def make_track_points(
    n: int = 5,
    start: str = "2020-01-01 12:00:00",
    freq: str = "min",
    *,
    time_audible: pd.Series | None = None,
) -> gpd.GeoDataFrame:
    """Minimal spline-like GeoDataFrame for segment / reload tests."""
    point_dt = pd.date_range(start, periods=n, freq=freq)
    if time_audible is None:
        time_audible = point_dt
    return gpd.GeoDataFrame(
        {"point_dt": point_dt, "time_audible": time_audible},
        geometry=[Point(float(i), float(i)) for i in range(n)],
        crs="EPSG:4326",
    )
