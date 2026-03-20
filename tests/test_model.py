import numpy as np
from model import predict, get_accuracy

def test_output_shape():
    """model must return one prediction per input row"""
    X = np.random.rand(10, 3)
    result = predict(X)
    assert result.shape == (10,), f"Wrong shape: {result.shape}"

def test_output_is_numbers():
    """model must return floats, not strings or None"""
    X = np.random.rand(5, 3)
    result = predict(X)
    assert result.dtype in [np.float32, np.float64]

def test_accuracy_above_threshold():
    """model accuracy must never drop below 85%"""
    acc = get_accuracy()
    assert acc >= 0.85, f"Accuracy too low: {acc:.2f}"