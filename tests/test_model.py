import numpy as np
from model import predict, get_accuracy

def test_output_shape():
    X = np.random.rand(10, 3)
    result = predict(X)
    assert result.shape == (10,)

def test_accuracy_above_threshold():
    acc = get_accuracy()
    assert acc >= 0.85
