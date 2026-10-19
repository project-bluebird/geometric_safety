"""Checks of the independent reference and production trace instrumentation."""

import mpmath as mp
import numpy as np
import pytest

from geometric_safety import relevant_aircraft as ra
from paper.run_turn_numerics import cases, traced_solver
from paper.turn_numerical_reference import basis, certificate, distance


def test_distance_handles_points_segments_and_interior() -> None:
    with mp.workdps(80):
        assert distance([(mp.mpf(3), mp.mpf(4))] * 4) == 5
        assert distance([(mp.mpf(-1), mp.mpf(2)), (mp.mpf(1), mp.mpf(2))]) == 2
        assert distance([(mp.mpf(-1), mp.mpf(-1)), (mp.mpf(1), mp.mpf(-1)), (mp.mpf(0), mp.mpf(1))]) == 0


@pytest.mark.parametrize(("angle", "rate", "t"), [(90.0, 3.0, 20.0), (-60.0, 2.0, 100.0), (1e-9, 1e-10, 600.0)])
def test_basis_matches_independent_quadrature(angle: float, rate: float, t: float) -> None:
    with mp.workdps(80):
        h, target, mp_rate, mp_time = map(mp.mpf, (12.0, 12.0 + angle, rate, t))
        duration = abs(target - h) / mp_rate
        signed_rate = mp.sign(target - h) * mp_rate
        points = sorted({mp.mpf(0), min(duration, mp_time), mp_time})
        for index, trig in enumerate((mp.sin, mp.cos)):
            result = mp.quad(lambda s, trig=trig: trig(mp.radians(h + signed_rate * min(s, duration))), points)
            assert abs(result - basis(h, target, mp_rate, mp_time)[index]) < mp.mpf("1e-55")


def test_reference_for_rigid_translation_and_constant_inflation() -> None:
    _, case = next(cases(1, 13))
    case.update(
        rel_pos0=np.array([9260.0, 0.0]),
        a_heading0_deg=0.0,
        a_target_heading_deg=90.0,
        b_heading0_deg=0.0,
        b_target_heading_deg=90.0,
        a_speed_kt=300.0,
        b_speed_kt=300.0,
        a_turn_rate_deg_sec=3.0,
        b_turn_rate_deg_sec=3.0,
        speed_diff_kt=0.0,
        eta_const_m=2.0,
    )
    with mp.workdps(80):
        result = certificate(case, 0.0, 60.0)
        assert abs(result["lower"] - 9260) < mp.mpf("1e-30")
        assert abs(result["residual"] + 2) < mp.mpf("1e-30")


def test_trace_matches_production_and_detects_guard_effect() -> None:
    solver, _ = traced_solver()
    trace = np.empty((2048, 6))
    for _, case in cases(16, 42):
        trace.fill(np.nan)
        assert solver(**case, study_trace=trace) == ra._catch_up_projection_interval_with_turns_local(
            **case,
            directional_mode=0,
        )
        rows = trace[~np.isnan(trace[:, 0])]
        assert len(rows) > 0
        assert np.all(rows[:, 1] >= rows[:, 0])
    case.update(
        rel_pos0=np.array([9260.0, 0.0]),
        a_heading0_deg=0.0,
        a_target_heading_deg=90.0,
        b_heading0_deg=0.0,
        b_target_heading_deg=90.0,
        a_speed_kt=300.0,
        b_speed_kt=300.0,
        a_turn_rate_deg_sec=3.0,
        b_turn_rate_deg_sec=3.0,
        speed_diff_kt=0.0,
        eta_const_m=0.0,
        eta_rate_mps=0.0,
        turn_delay_max_s=0.0,
        turn_rate_error_max=0.0,
        turn_speed_schedule_uncertainty_kt=0.0,
        projection_time_s=60.0,
    )
    assert solver(**case, study_trace=trace, study_guard=0.0)[0]
    assert not solver(**case, study_trace=trace, study_guard=1e-6)[0]
