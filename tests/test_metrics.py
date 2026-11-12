import numpy as np
import pytest

from finsent_calib.metrics import brier_score, expected_calibration_error, negative_log_likelihood

# Two sentences at confidence 0.75 and two at 0.55; in each pair one is right and one wrong.
PROBS = np.array(
    [
        [0.75, 0.15, 0.10],
        [0.75, 0.15, 0.10],
        [0.20, 0.55, 0.25],
        [0.20, 0.55, 0.25],
    ]
)
Y_TRUE = np.array([0, 1, 1, 2])


def test_ece_matches_hand_calculation() -> None:
    # bin 0.75: share 0.5, |0.5 - 0.75| = 0.25; bin 0.55: share 0.5, |0.5 - 0.55| = 0.05
    expected = 0.5 * 0.25 + 0.5 * 0.05
    assert expected_calibration_error(PROBS, Y_TRUE) == pytest.approx(expected)


def test_ece_is_zero_when_confidence_equals_accuracy() -> None:
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    y_true = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, y_true) == pytest.approx(0.0)


def test_nll_matches_hand_calculation() -> None:
    expected = -(np.log(0.75) + np.log(0.15) + np.log(0.55) + np.log(0.25)) / 4
    assert negative_log_likelihood(PROBS, Y_TRUE) == pytest.approx(expected)


def test_perfect_prediction_has_zero_nll_and_brier() -> None:
    y_true = np.array([0, 1, 2])
    probs = np.eye(3)
    assert negative_log_likelihood(probs, y_true) == pytest.approx(0.0)
    assert brier_score(probs, y_true) == pytest.approx(0.0)


def test_brier_of_uniform_guess_is_two_thirds() -> None:
    probs = np.full((3, 3), 1 / 3)
    y_true = np.array([0, 1, 2])
    assert brier_score(probs, y_true) == pytest.approx(2 / 3)
