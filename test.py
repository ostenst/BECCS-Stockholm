from controller import evaluate_model
from unittest.mock import MagicMock


def test_evaluate_model():

    model = MagicMock()
    model.uncertainties = [MagicMock()]
    results = evaluate_model(model)
    assert len(results) == 10000
    assert isinstance(results, list)
