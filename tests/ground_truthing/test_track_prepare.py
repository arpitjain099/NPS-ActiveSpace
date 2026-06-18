import datetime as dt

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.dates import date2num, num2date

from nps_active_space.ground_truthing.track_prepare import (
    audible_ranges_from_annotations,
    limit_line_bounds,
    load_spectrogram,
)
from nps_active_space.utils.computation import interpolate_spline
from helpers import make_track_points


class TestInterpolateSpline:
    def test_duplicate_point_dt_raises_clear_error(self):
        points = make_track_points(4)
        points["point_dt"] = [
            pd.Timestamp("2025-01-07 05:00:00"),
            pd.Timestamp("2025-01-07 05:00:00"),
            pd.Timestamp("2025-01-07 05:01:00"),
            pd.Timestamp("2025-01-07 05:02:00"),
        ]
        points["z"] = 0.0
        with pytest.raises(ValueError, match="unique timestamp"):
            interpolate_spline(points)


class TestLoadSpectrogram:
    def test_slices_naive_nvspl_with_site_local_points(self):
        nvspl_index = pd.date_range("2025-01-07 04:00", periods=180, freq="min")
        nvspl = pd.DataFrame(
            {"12.5": 40.0, "20000": 30.0},
            index=nvspl_index,
        )
        point_dt = pd.date_range("2025-01-07 05:00", periods=6, freq="min")
        points = make_track_points(6)
        points["point_dt"] = point_dt
        spectro = load_spectrogram(nvspl, points)
        assert not spectro.empty
        assert spectro.index.min() >= nvspl_index[0]
        assert spectro.index.max() <= nvspl_index[-1]


class TestLimitLineBounds:
    def test_centered_window_within_spectrogram(self):
        closest = pd.Timestamp("2020-01-01 12:05:00")
        x_lims = date2num(pd.date_range("2020-01-01 12:00", periods=11, freq="min"))
        lower, upper = limit_line_bounds(closest, x_lims)
        assert lower == date2num(closest - dt.timedelta(seconds=60))
        assert upper == date2num(closest + dt.timedelta(seconds=60))

    def test_clamps_to_spectrogram_edges(self):
        closest = pd.Timestamp("2020-01-01 12:00:30")
        x_lims = date2num(pd.date_range("2020-01-01 12:00", periods=3, freq="min"))
        lower, upper = limit_line_bounds(closest, x_lims)
        assert lower == x_lims[0]
        assert upper == date2num(closest + dt.timedelta(seconds=60))

    def test_upper_clamped_at_end(self):
        closest = pd.Timestamp("2020-01-01 12:09:30")
        x_lims = date2num(pd.date_range("2020-01-01 12:00", periods=10, freq="min"))
        lower, upper = limit_line_bounds(closest, x_lims)
        assert upper == x_lims[-1]
        assert lower == date2num(closest - dt.timedelta(seconds=60))


class TestAudibleRangesFromAnnotations:
    def test_empty_annotations_returns_default_slider_range(self):
        spline = make_track_points(5)
        lower = date2num(spline.point_dt.iat[2] - dt.timedelta(seconds=30))
        upper = date2num(spline.point_dt.iat[2] + dt.timedelta(seconds=30))
        annots = gpd.GeoDataFrame(
            columns=["valid", "audible", "start_dt", "end_dt"],
            geometry=[],
            crs="EPSG:4326",
        )
        result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert len(result) == 1
        assert result[0][0] == num2date(lower)
        assert result[0][1] == num2date(upper)

    def test_restores_audible_range_on_time_audible_axis(self):
        spline = make_track_points(6)
        start_dt = spline.point_dt.iat[1]
        end_dt = spline.point_dt.iat[4]
        annots = gpd.GeoDataFrame(
            {
                "valid": [True],
                "audible": [True],
                "start_dt": [start_dt],
                "end_dt": [end_dt],
            },
            geometry=[spline.geometry.iat[0]],
            crs="EPSG:4326",
        )
        lower = date2num(spline.point_dt.iat[0])
        upper = date2num(spline.point_dt.iat[-1])
        result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result == [
            [spline.time_audible.iat[1], spline.time_audible.iat[4]],
        ]

    def test_ignores_invalid_or_inaudible_rows(self):
        spline = make_track_points(5)
        annots = gpd.GeoDataFrame(
            {
                "valid": [False, True],
                "audible": [True, False],
                "start_dt": [spline.point_dt.iat[0], spline.point_dt.iat[1]],
                "end_dt": [spline.point_dt.iat[2], spline.point_dt.iat[3]],
            },
            geometry=[spline.geometry.iat[0], spline.geometry.iat[1]],
            crs="EPSG:4326",
        )
        lower = date2num(spline.point_dt.iat[0])
        upper = date2num(spline.point_dt.iat[-1])
        result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result == []

    def test_maps_annotations_onto_time_audible_axis(self):
        point_dt = pd.date_range("2020-01-01 12:00", periods=6, freq="min")
        time_audible = point_dt + pd.Timedelta(seconds=45)
        spline = make_track_points(6, time_audible=time_audible)
        annots = gpd.GeoDataFrame(
            {
                "valid": [True],
                "audible": [True],
                "start_dt": [point_dt[2]],
                "end_dt": [point_dt[4]],
            },
            geometry=[spline.geometry.iat[0]],
            crs="EPSG:4326",
        )
        lower = date2num(point_dt[0])
        upper = date2num(point_dt[-1])
        result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result == [[time_audible[2], time_audible[4]]]

    def test_returns_none_when_start_outside_tolerance(self):
        spline = make_track_points(5)
        annots = gpd.GeoDataFrame(
            {
                "valid": [True],
                "audible": [True],
                "start_dt": [spline.point_dt.iat[0] - pd.Timedelta(seconds=5)],
                "end_dt": [spline.point_dt.iat[2]],
            },
            geometry=[spline.geometry.iat[0]],
            crs="EPSG:4326",
        )
        lower = date2num(spline.point_dt.iat[0])
        upper = date2num(spline.point_dt.iat[-1])
        with pytest.warns(UserWarning, match="annotation start"):
            result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result is None

    def test_returns_none_when_end_outside_tolerance(self):
        spline = make_track_points(5)
        annots = gpd.GeoDataFrame(
            {
                "valid": [True],
                "audible": [True],
                "start_dt": [spline.point_dt.iat[1]],
                "end_dt": [spline.point_dt.iat[-1] + pd.Timedelta(seconds=5)],
            },
            geometry=[spline.geometry.iat[0]],
            crs="EPSG:4326",
        )
        lower = date2num(spline.point_dt.iat[0])
        upper = date2num(spline.point_dt.iat[-1])
        with pytest.warns(UserWarning, match="annotation end"):
            result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result is None

    def test_accepts_subsecond_annotation_drift(self):
        spline = make_track_points(5)
        annots = gpd.GeoDataFrame(
            {
                "valid": [True],
                "audible": [True],
                "start_dt": [spline.point_dt.iat[1] + pd.Timedelta(milliseconds=500)],
                "end_dt": [spline.point_dt.iat[3] - pd.Timedelta(milliseconds=500)],
            },
            geometry=[spline.geometry.iat[0]],
            crs="EPSG:4326",
        )
        lower = date2num(spline.point_dt.iat[0])
        upper = date2num(spline.point_dt.iat[-1])
        result = audible_ranges_from_annotations(annots, spline, lower, upper)
        assert result == [
            [spline.time_audible.iat[1], spline.time_audible.iat[3]],
        ]
