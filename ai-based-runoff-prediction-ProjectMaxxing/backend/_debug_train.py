import traceback
from app.schemas import TrainAutoRequest
from app.services.training_service import run_auto_training

payload = TrainAutoRequest(
    train_end_year=2000,
    test_start_year=2006,
    search_type="randomized",
    n_iter=1,
)

print("starting")
try:
    result = run_auto_training(payload)
    print("ok", result.keys())
except Exception as exc:
    print(type(exc).__name__, exc)
    traceback.print_exc()
