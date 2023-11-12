import torch
import numpy as np
import pytest

import poly_pitch_net as ppn


@pytest.fixture(scope="session", autouse=True)
def metrics_config():
    target_hz = torch.tensor([82.9, 300, 256.5, 110, 192.6, 142, 456, 604.9])
    target_cents = ppn.tools.frequency_to_cents(target_hz)

    # threshold in cents, double that to get the total threshold
    threshold = 20
    noise = torch.tensor(np.random.normal(
        loc=threshold, 
        scale=10, 
        size=target_cents.shape))

    pred_cents = target_cents + noise

    return target_cents, pred_cents, threshold


def test_accuracy(metrics_config):
    target_cents, pred_cents, threshold = metrics_config

    true_with_threshold = pred_cents[abs(target_cents - pred_cents) <= threshold]

    accuracy_score = true_with_threshold.numel() / pred_cents.numel() 

    print(f"accuracy score {accuracy_score}")

    accuracy = ppn.evaluate.metrics.Accuracy(threshold)
    accuracy.update(pred_cents, target_cents)

    assert accuracy() == accuracy_score


def test_rmse(metrics_config):
    target_cents, pred_cents, _ = metrics_config

    rmse = ppn.evaluate.metrics.RMSE()
    rmse.update(pred_cents, target_cents)

    print(rmse())


def test_rpa(metrics_config):
    target_cents, pred_cents, threshold = metrics_config

    rpa = ppn.evaluate.metrics.RPA(threshold)
    rpa.update(pred_cents, target_cents)

    print(rpa())
