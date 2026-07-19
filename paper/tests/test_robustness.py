"""Tests for the robustness-study helpers in paper/robustness.py."""

import math

import numpy as np
import pytest

from geometric_safety.util import DEG_TO_RAD, KT_TO_MPS, NMI_TO_M
from paper.evaluation import DEFAULT_REF_LAT, DEFAULT_REF_LON, Encounter
from paper.robustness import (
    PerturbationBounds,
    TruthModelConfig,
    deviation_envelope_m,
    evaluate_robustness_case,
    simulate_truth_min_separation,
    solver_envelope_kwargs,
)


def _straight_crossing(separation_nmi: float, horizon_s: float = 300.0, speed_diff_kt: float = 15.0) -> Encounter:
    return Encounter(
        rel_east_m=separation_nmi * NMI_TO_M,
        rel_north_m=0.0,
        a_heading0_deg=0.0,
        a_target_heading_deg=0.0,
        a_speed_kt=260.0,
        a_turn_rate_deg_sec=0.0,
        b_heading0_deg=0.0,
        b_target_heading_deg=0.0,
        b_speed_kt=250.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=speed_diff_kt,
        projection_time_s=horizon_s,
    )


def _turning_encounter(separation_nmi: float, turn_deg: float, horizon_s: float = 300.0) -> Encounter:
    return Encounter(
        rel_east_m=separation_nmi * NMI_TO_M,
        rel_north_m=-6.0 * NMI_TO_M,
        a_heading0_deg=0.0,
        a_target_heading_deg=turn_deg,
        a_speed_kt=260.0,
        a_turn_rate_deg_sec=3.0,
        b_heading0_deg=180.0,
        b_target_heading_deg=180.0,
        b_speed_kt=230.0,
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=15.0,
        projection_time_s=horizon_s,
    )


class TestEnvelopeConversion:
    def test_solver_envelope_kwargs_formula(self):
        encounter = _straight_crossing(12.0)
        bounds = PerturbationBounds(differential_wind_kt=5.0, position_error_m=120.0, heading_error_deg=1.0)
        params = solver_envelope_kwargs(encounter, bounds)

        a_max = (260.0 + 15.0) * KT_TO_MPS
        b_max = (250.0 + 15.0) * KT_TO_MPS
        expected_rate = 5.0 * KT_TO_MPS + (1.0 * DEG_TO_RAD) * (a_max + b_max)
        assert params["eta_const_m"] == pytest.approx(120.0)
        assert params["eta_rate_mps"] == pytest.approx(expected_rate, rel=1e-12)

    def test_deviation_envelope_linear_terms(self):
        encounter = _straight_crossing(12.0)
        bounds = PerturbationBounds(differential_wind_kt=4.0, position_error_m=90.0)
        eta_at_100 = deviation_envelope_m(encounter, bounds, 100.0)
        expected = 90.0 + 4.0 * KT_TO_MPS * 100.0
        assert eta_at_100 == pytest.approx(expected, rel=1e-12)

    def test_null_bounds_have_zero_envelope(self):
        encounter = _turning_encounter(10.0, 60.0)
        assert PerturbationBounds().is_null()
        assert deviation_envelope_m(encounter, PerturbationBounds(), 300.0) == pytest.approx(0.0)


class TestTruthSimulation:
    def test_null_bounds_match_analytic_straight_geometry(self):
        encounter = _straight_crossing(8.0, speed_diff_kt=0.0)
        rng = np.random.default_rng(7)
        sim_min = simulate_truth_min_separation(encounter, PerturbationBounds(), rng)

        # With no perturbations and a degenerate speed interval the truth trajectory
        # is the nominal one: both aircraft northbound, closing at 10 kt from 8 NMI
        # east offset, so the separation stays 8 NMI east and shrinks only north.
        rel_north_final = (260.0 - 250.0) * KT_TO_MPS * 300.0
        expected = math.hypot(8.0 * NMI_TO_M, 0.0)
        assert rel_north_final < 8.0 * NMI_TO_M  # sanity: north drift stays small
        assert sim_min == pytest.approx(expected, abs=0.5)

    def test_schedule_draw_respects_interval_clipping(self):
        bounds = PerturbationBounds(schedule_uncertainty_kt=40.0)
        rng = np.random.default_rng(3)
        min_speed = max(260.0 - 15.0, 0.0) * KT_TO_MPS
        max_speed = (260.0 + 15.0) * KT_TO_MPS
        from paper.robustness import _sample_aircraft_draw

        for _ in range(50):
            draw = _sample_aircraft_draw(
                rng,
                speed_kt=260.0,
                speed_diff_kt=15.0,
                heading0_deg=0.0,
                target_heading_deg=60.0,
                turn_rate_deg_sec=3.0,
                bounds=bounds,
                adversarial=True,
                truth=TruthModelConfig(),
                horizon_s=300.0,
            )
            assert draw.schedule_speeds_mps.size > 0
            assert float(draw.schedule_speeds_mps.min()) >= min_speed - 1e-9
            assert float(draw.schedule_speeds_mps.max()) <= max_speed + 1e-9

    def test_wind_shifts_min_separation_at_most_linearly(self):
        encounter = _straight_crossing(8.0)
        bounds = PerturbationBounds(differential_wind_kt=5.0)
        rng = np.random.default_rng(11)
        for _ in range(20):
            sim_min = simulate_truth_min_separation(encounter, bounds, rng)
            eta_horizon = deviation_envelope_m(encounter, bounds, encounter.projection_time_s)
            baseline = simulate_truth_min_separation(encounter, PerturbationBounds(), np.random.default_rng(11))
            assert sim_min >= baseline - eta_horizon - 1.0


class TestTruthModelConfig:
    def test_default_config_is_in_class(self):
        assert TruthModelConfig().is_in_class()
        assert not TruthModelConfig(exceedance_factor=1.5).is_in_class()
        assert not TruthModelConfig(ou_speed_sigma_kt=5.0).is_in_class()
        assert not TruthModelConfig(roll_time_s=5.0).is_in_class()
        assert not TruthModelConfig(heading_jitter_deg=0.5).is_in_class()

    def test_default_truth_matches_legacy_behaviour(self):
        encounter = _turning_encounter(9.0, 45.0)
        bounds = PerturbationBounds(differential_wind_kt=3.0, schedule_uncertainty_kt=8.0)
        base = simulate_truth_min_separation(encounter, bounds, np.random.default_rng(5))
        with_config = simulate_truth_min_separation(
            encounter, bounds, np.random.default_rng(5), truth=TruthModelConfig()
        )
        assert with_config == pytest.approx(base, rel=1e-12)

    def test_ou_speeds_stay_inside_interval(self):
        from paper.robustness import _ou_speed_path_mps

        rng = np.random.default_rng(9)
        min_speed = 240.0 * KT_TO_MPS
        max_speed = 280.0 * KT_TO_MPS
        _, values = _ou_speed_path_mps(
            rng,
            constant_speed_mps=260.0 * KT_TO_MPS,
            min_speed_mps=min_speed,
            max_speed_mps=max_speed,
            sigma_kt=25.0,
            timescale_s=30.0,
            horizon_s=900.0,
        )
        assert float(values.min()) >= min_speed - 1e-9
        assert float(values.max()) <= max_speed + 1e-9

    def test_roll_dynamics_change_the_path(self):
        encounter = _turning_encounter(9.0, 60.0)
        base = simulate_truth_min_separation(
            encounter, PerturbationBounds(), np.random.default_rng(13), truth=TruthModelConfig()
        )
        rolled = simulate_truth_min_separation(
            encounter, PerturbationBounds(), np.random.default_rng(13), truth=TruthModelConfig(roll_time_s=6.0)
        )
        assert rolled != pytest.approx(base, abs=1e-6)

    def test_straight_ou_drift_never_violates_plain_certificate(self):
        """Proposition 2.1 in action: whole-horizon speed variation is free when straight.

        For straight-heading pairs, any measurable speed schedule inside the interval
        reaches only positions the constant-speed hull already covers, so even the
        uninflated certificate must survive OU speed drift.
        """
        truth = TruthModelConfig(ou_speed_sigma_kt=20.0, ou_speed_timescale_s=45.0)
        rng = np.random.default_rng(20260720)
        checked = 0
        for separation_nmi in (7.0, 9.0, 12.0):
            encounter = _straight_crossing(separation_nmi)
            result = evaluate_robustness_case(
                encounter,
                PerturbationBounds(),
                n_sims=40,
                rng=rng,
                ref_lat=DEFAULT_REF_LAT,
                ref_lon=DEFAULT_REF_LON,
                adversarial=True,
                truth=truth,
            )
            if result.certified_safe:
                checked += 1
                assert result.n_below_threshold == 0
        assert checked > 0


class TestCertificateValidity:
    @pytest.mark.parametrize(
        "bounds",
        [
            PerturbationBounds(differential_wind_kt=5.0),
            PerturbationBounds(position_error_m=250.0),
            PerturbationBounds(heading_error_deg=1.5),
            PerturbationBounds(turn_delay_max_s=8.0, turn_rate_error_max=0.15),
            PerturbationBounds(schedule_uncertainty_kt=12.0),
        ],
        ids=["wind", "position", "heading", "execution", "schedule"],
    )
    def test_inflated_certificate_never_violated(self, bounds):
        """Certified-safe under inflation must survive adversarial truth simulation."""
        encounters = [
            _straight_crossing(8.0),
            _straight_crossing(12.0),
            _turning_encounter(9.0, 45.0),
            _turning_encounter(12.0, 60.0),
            _turning_encounter(7.5, 30.0),
        ]
        rng = np.random.default_rng(20260719)
        any_certified = False
        for encounter in encounters:
            result = evaluate_robustness_case(
                encounter,
                bounds,
                n_sims=60,
                rng=rng,
                ref_lat=DEFAULT_REF_LAT,
                ref_lon=DEFAULT_REF_LON,
                adversarial=True,
            )
            any_certified = any_certified or result.inflated_certified_safe
            assert result.n_violations == 0
        assert any_certified, "test suite should contain at least one certifiable encounter"
