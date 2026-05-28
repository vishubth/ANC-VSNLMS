import numpy as np

from model.vsnlms import VSNLMS


def test_weight_update_executes() -> None:
    filter_instance = VSNLMS(
        mu=0.01,
        mu_max=0.1,
        mu_min=0.0001,
        m0=10,
        m1=10,
        alpha=1.01,
    )

    filter_instance.weights = np.zeros(8)

    input_signal = np.random.randn(8)

    error, updated_weights = filter_instance.update_weights(
        input_signal,
        desired_signal=1.0,
    )

    assert isinstance(error, float)
    assert updated_weights.shape[0] == 8


def test_weight_stabilization() -> None:
    filter_instance = VSNLMS(
        mu=0.01,
        mu_max=0.1,
        mu_min=0.0001,
        m0=10,
        m1=10,
        alpha=1.01,
    )

    filter_instance.weights = np.ones(8) * 100

    filter_instance._stabilize_weights()

    assert np.linalg.norm(filter_instance.weights) <= 10
