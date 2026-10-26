"""The public unicycle maneuvers exercise flat heading and rest boundaries."""

from unicycle import run_scenarios


def test_public_unicycle_scenarios():
    reports = {report["name"]: report for report in run_scenarios()}
    assert set(reports) == {"straight_start_stop", "lane_change", "corner_90", "u_turn"}
    for report in reports.values():
        assert report["wheel_speed_violation"] < 1e-9
        assert abs(report["start_heading_error"]) < 1e-5
        assert abs(report["end_heading_error"]) < 1e-5
        assert report["final_position_error"] < 1e-9
    assert reports["straight_start_stop"]["max_abs_yaw_rate"] == 0.0
    assert abs(reports["lane_change"]["heading_change"]) < 1e-9
    assert reports["corner_90"]["heading_change"] > 1.3
    assert reports["u_turn"]["heading_change"] > 2.8
