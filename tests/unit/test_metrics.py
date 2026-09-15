import numpy as np
import pytest

from defectfirst.evaluation.calibration import calibrate, conservative_threshold
from defectfirst.evaluation.metrics import aupro, image_score, metrics


@pytest.mark.parametrize("limit", [0.05, 0.30, 1.0])
def test_aupro_analytic_perfect_reverse_constant(limit):
    truth = np.array([[1, 0], [0, 0]])
    assert aupro([truth], [truth.astype(float)], limit) == pytest.approx(1)
    assert aupro([truth], [1 - truth.astype(float)], limit) == pytest.approx(0)
    assert aupro([truth], [np.ones_like(truth, dtype=float)], limit) == pytest.approx(limit / 2)


def test_aupro_region_macro_is_not_pixel_recall():
    truth = np.zeros((4, 5), int)
    truth[0, 0] = 1
    truth[2:, 2:] = 1
    prediction = np.zeros_like(truth, float)
    prediction[0, 0] = 1
    # First region is perfect, large region is not; the first plateau is 0.5.
    assert aupro([truth], [prediction], 0.3) == pytest.approx(0.5 + 0.25 * 0.3)


def test_all_normal_pixels_are_counted_and_ap_matches_manual_precision():
    truths = [np.array([[1, 0], [1, 0]]), np.zeros((1, 2), int)]
    scores = [np.array([[0.9, 0.8], [0.7, 0.1]]), np.array([[0.6, 0.2]])]
    result = metrics(truths, scores, 0.5, 0.5)
    assert result["pixel_ap"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["pixels"] == 6 and result["fp"] == 2 and result["tp"] == 2
    assert result["normal_image_fpr"] == 1


def test_calibration_ties_conservative_and_per_image_weighting():
    assert conservative_threshold(np.array([1, 1, 2]), np.ones(3), 0.2) == 2
    small, large = np.array([[0.9]]), np.zeros((100, 100))
    calibrated = calibrate([small, large], pixel_fpr=0.1)
    assert calibrated["tau_pix"] == 0.9  # Large normal images do not swamp the small image.
    assert calibrated["observed_pixel_fpr"] <= 0.1


def test_empty_and_nonfinite_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        calibrate([])
    with pytest.raises(ValueError):
        metrics([np.zeros((1, 1))], [np.array([[np.nan]])], 0.5, 0.5)
    assert aupro([np.zeros((1, 1))], [np.zeros((1, 1))], 0.3) is None
    assert image_score(np.array([[0.1, 0.8]])) == 0.8
