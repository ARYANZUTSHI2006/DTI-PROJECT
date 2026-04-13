import json
from app.schemas import TrainAutoRequest
from app.services.training_service import load_training_dataset, _invoke_auto_train

payload = TrainAutoRequest(train_end_year=2000, test_start_year=2006, search_type="randomized", n_iter=1)
df = load_training_dataset()
result = _invoke_auto_train(df, payload)
print(type(result))
if isinstance(result, dict):
    print(sorted(result.keys()))
    for key in ["best_model", "all_models", "y_true", "y_test", "y_pred", "predictions", "model", "best_estimator", "X", "X_test"]:
        if key in result:
            val = result[key]
            print(key, type(val), getattr(val, "shape", None))
