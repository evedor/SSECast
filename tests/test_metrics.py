import numpy as np
from ssecast.metrics import anomaly_correlation_np

def test_anomaly_correlation_is_one_for_identical_fields():
    truth = np.array([[1.0, 3.0], [2.0, 5.0]])
    climatology = np.array([0.0, 0.0])
    assert np.isclose(anomaly_correlation_np(truth, truth, climatology), 1.0)
