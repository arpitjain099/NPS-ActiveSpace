import datetime as dt

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from nps_active_space.ground_truthing.segments import (
    build_annotation_segments,
    collapse_audible_ranges,
)
from helpers import make_track_points


class TestCollapseAudibleRanges:
    def test_merges_overlap(self):
        t0 = dt.datetime(2020, 1, 1, 12, 0, 0)
        ranges = [
            [t0, t0 + dt.timedelta(minutes=5)],
            [t0 + dt.timedelta(minutes=3), t0 + dt.timedelta(minutes=10)],
        ]
        out = collapse_audible_ranges(ranges)
        assert len(out) == 1
        assert out[0] == [t0, t0 + dt.timedelta(minutes=10)]

    def test_keeps_separate_non_overlapping(self):
        t0 = dt.datetime(2020, 1, 1, 12, 0, 0)
        ranges = [
            [t0, t0 + dt.timedelta(minutes=2)],
            [t0 + dt.timedelta(minutes=5), t0 + dt.timedelta(minutes=7)],
        ]
        out = collapse_audible_ranges(ranges)
        assert len(out) == 2
        assert out[0] == ranges[0]
        assert out[1] == ranges[1]

    def test_single_range_unchanged(self):
        t0 = dt.datetime(2020, 1, 1, 12, 0, 0)
        ranges = [[t0, t0 + dt.timedelta(minutes=3)]]
        assert collapse_audible_ranges(ranges) == ranges


class TestBuildAnnotationSegments:
    def test_all_inaudible_when_no_ranges(self):
        points = make_track_points(3)
        result = build_annotation_segments("T1", points, audible_ranges=[], valid=True)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["audible"] == False
        assert row["valid"] == True
        assert row["_id"] == "T1"
        assert row["start_dt"] == points.point_dt.iat[0]
        assert row["end_dt"] == points.point_dt.iat[-1]

    def test_invalid_track_single_segment(self):
        points = make_track_points(4)
        result = build_annotation_segments(
            "T1", points, audible_ranges=[[points.point_dt.iat[1], points.point_dt.iat[2]]], valid=False
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["valid"] == False
        assert row["audible"] == False
        assert row["start_dt"] == points.point_dt.iat[0]
        assert row["end_dt"] == points.point_dt.iat[-1]

    def test_tail_inaudible_after_audible_window(self):
        points = make_track_points(6)
        t = points.point_dt
        audible_ranges = [[t.iat[1], t.iat[3]]]
        result = build_annotation_segments("T1", points, audible_ranges=audible_ranges)
        audible = result[result["audible"]]
        inaudible = result[~result["audible"]]
        assert len(audible) == 1
        assert len(inaudible) == 1
        assert audible.iloc[0]["start_dt"] == t.iat[1]
        assert audible.iloc[0]["end_dt"] == t.iat[2]
        assert inaudible.iloc[0]["start_dt"] == t.iat[3]
        assert inaudible.iloc[0]["end_dt"] == t.iat[4]
        assert all(result["valid"])
        assert isinstance(audible.iloc[0].geometry, LineString)

    def test_two_audible_windows_with_gap(self):
        points = make_track_points(8)
        t = points.point_dt
        audible_ranges = [[t.iat[1], t.iat[3]], [t.iat[5], t.iat[7]]]
        result = build_annotation_segments("T1", points, audible_ranges=audible_ranges)
        assert len(result[result["audible"]]) == 2
        assert len(result[~result["audible"]]) == 1
        gap = result[~result["audible"]].iloc[0]
        assert gap["start_dt"] == t.iat[3]
        assert gap["end_dt"] == t.iat[4]

    def test_overlapping_input_ranges_collapsed_before_split(self):
        points = make_track_points(6)
        t = points.point_dt
        audible_ranges = [[t.iat[1], t.iat[3]], [t.iat[2], t.iat[4]]]
        result = build_annotation_segments("T1", points, audible_ranges=audible_ranges)
        audible = result[result["audible"]]
        assert len(audible) == 1
        assert audible.iloc[0]["start_dt"] == t.iat[1]
        assert audible.iloc[0]["end_dt"] == t.iat[3]

    def test_splits_on_time_audible_not_point_dt(self):
        point_dt = pd.date_range("2020-01-01 12:00", periods=6, freq="min")
        time_audible = point_dt + pd.Timedelta(seconds=30)
        points = make_track_points(6, time_audible=time_audible)
        # Audible window on time_audible axis between 2nd and 4th samples.
        audible_ranges = [[time_audible[1], time_audible[3]]]
        result = build_annotation_segments("T1", points, audible_ranges=audible_ranges)
        audible = result[result["audible"]].iloc[0]
        assert audible["start_dt"] == point_dt[1]
        assert audible["end_dt"] == point_dt[2]

    def test_note_propagates_to_segments(self):
        points = make_track_points(5)
        t = points.point_dt
        result = build_annotation_segments(
            "T1",
            points,
            audible_ranges=[[t.iat[1], t.iat[3]]],
            note="test note",
        )
        assert result["note"].eq("test note").all()
