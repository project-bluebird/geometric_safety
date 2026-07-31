"""Tests for the independent high-precision straight-heading oracle."""

import dataclasses
import math

import numpy as np
import pytest

from geometric_safety.util import KT_TO_MPS, NMI_TO_M
from paper.evaluation import Encounter, evaluate_straight_oracle
from paper.independent_oracle import independent_straight_outcome


def _outcome(encounter: Encounter):
    return independent_straight_outcome(
        rel_east_m=encounter.rel_east_m,
        rel_north_m=encounter.rel_north_m,
        a_heading_deg=encounter.a_heading0_deg,
        a_speed_kt=encounter.a_speed_kt,
        b_heading_deg=encounter.b_heading0_deg,
        b_speed_kt=encounter.b_speed_kt,
        speed_diff_kt=encounter.speed_diff_kt,
        horizon_s=encounter.projection_time_s,
        threshold_m=encounter.separation_threshold_m,
    )


def _straight_encounter(rng: np.random.Generator) -> Encounter:
    bearing = rng.uniform(0.0, 2.0 * math.pi)
    separation_m = rng.uniform(6.0, 40.0) * NMI_TO_M
    return Encounter(
        rel_east_m=separation_m * math.sin(bearing),
        rel_north_m=separation_m * math.cos(bearing),
        a_heading0_deg=rng.uniform(0.0, 360.0),
        a_target_heading_deg=0.0,
        a_speed_kt=rng.uniform(220.0, 450.0),
        a_turn_rate_deg_sec=0.0,
        b_heading0_deg=rng.uniform(0.0, 360.0),
        b_target_heading_deg=0.0,
        b_speed_kt=rng.uniform(220.0, 450.0),
        b_turn_rate_deg_sec=0.0,
        speed_diff_kt=rng.uniform(5.0, 30.0),
        projection_time_s=float(rng.choice(np.asarray([300.0, 600.0, 900.0, 1200.0]))),
    )


class TestAgainstProductionClosedForm:
    def test_random_encounters_agree_with_production(self):
        """Two independent implementations must agree to numerical precision."""
        rng = np.random.default_rng(20260731)
        decision_disagreements = 0
        for _ in range(200):
            encounter = _straight_encounter(rng)
            # Fix the target headings to the initial ones so the encounter is
            # genuinely straight for both code paths.
            encounter = dataclasses.replace(
                encounter,
                a_target_heading_deg=encounter.a_heading0_deg,
                b_target_heading_deg=encounter.b_heading0_deg,
            )
            production = evaluate_straight_oracle(encounter)
            indep_safe, indep_distance_m, indep_time_s = _outcome(encounter)

            assert production.min_distance_m == pytest.approx(indep_distance_m, abs=1e-4)
            assert production.closest_time_s == pytest.approx(indep_time_s, abs=1e-4)
            if bool(production.is_safe) != indep_safe:
                # A decision flip is only legitimate within numerical distance
                # of the threshold itself.
                assert abs(indep_distance_m - encounter.separation_threshold_m) < 1e-4
                decision_disagreements += 1
        assert decision_disagreements == 0

    def test_constant_separation_parallel_tracks(self):
        """Equal-speed parallel tracks keep their initial separation forever."""
        encounter = Encounter(
            rel_east_m=8.0 * NMI_TO_M,
            rel_north_m=0.0,
            a_heading0_deg=0.0,
            a_target_heading_deg=0.0,
            a_speed_kt=250.0,
            a_turn_rate_deg_sec=0.0,
            b_heading0_deg=0.0,
            b_target_heading_deg=0.0,
            b_speed_kt=250.0,
            b_turn_rate_deg_sec=0.0,
            speed_diff_kt=0.0,
            projection_time_s=600.0,
        )
        indep_safe, indep_distance_m, indep_time_s = _outcome(encounter)
        assert indep_safe
        assert indep_distance_m == pytest.approx(8.0 * NMI_TO_M, rel=1e-12)
        assert indep_time_s == pytest.approx(0.0, abs=1e-9)

    def test_catch_up_reaches_co_location(self):
        """A faster follower on the same track can close to zero within the horizon."""
        gap_m = 5.0 * NMI_TO_M
        encounter = Encounter(
            rel_east_m=0.0,
            rel_north_m=-gap_m,
            a_heading0_deg=0.0,
            a_target_heading_deg=0.0,
            a_speed_kt=300.0,
            a_turn_rate_deg_sec=0.0,
            b_heading0_deg=0.0,
            b_target_heading_deg=0.0,
            b_speed_kt=240.0,
            b_turn_rate_deg_sec=0.0,
            speed_diff_kt=10.0,
            projection_time_s=600.0,
        )
        indep_safe, indep_distance_m, indep_time_s = _outcome(encounter)
        assert not indep_safe
        assert indep_distance_m == pytest.approx(0.0, abs=1e-6)
        # Earliest co-location uses the fastest admissible closing speed.
        closing_mps = (300.0 + 10.0 - (240.0 - 10.0)) * KT_TO_MPS
        assert indep_time_s == pytest.approx(gap_m / closing_mps, rel=1e-6)

    def test_diverging_tracks_min_at_start(self):
        """Opening geometry has its minimum at time zero with earliest time zero."""
        encounter = Encounter(
            rel_east_m=10.0 * NMI_TO_M,
            rel_north_m=0.0,
            a_heading0_deg=90.0,
            a_target_heading_deg=90.0,
            a_speed_kt=300.0,
            a_turn_rate_deg_sec=0.0,
            b_heading0_deg=270.0,
            b_target_heading_deg=270.0,
            b_speed_kt=300.0,
            b_turn_rate_deg_sec=0.0,
            speed_diff_kt=20.0,
            projection_time_s=600.0,
        )
        indep_safe, indep_distance_m, indep_time_s = _outcome(encounter)
        assert indep_safe
        assert indep_distance_m == pytest.approx(10.0 * NMI_TO_M, rel=1e-12)
        assert indep_time_s == pytest.approx(0.0, abs=1e-9)
