# W&B Analysis Helper — Extract steady-state throughput

This helper shows how to pull a run’s recent step throughput from W&B to compute a steady‑state median.

## Python snippet

Use in any environment with `wandb` and `pandas` installed.

```python
import wandb
import pandas as pd
from datetime import timedelta

# Replace with your run path
run_path = "jajireen1-hrm_project/Sudoku-extreme-1k-aug-1000 ACT-torch/runs/n3zevn3d".replace(" ", "%20")

api = wandb.Api()
run = api.run(run_path)

# Pull history for key metrics (limit can be adjusted)
hist = run.history(pandas=True, keys=["train/steps_per_second", "train/step_time_s", "system/gpu_util_percent", "_timestamp"], samples=10000)

# Convert timestamp
hist["datetime"] = pd.to_datetime(hist["_timestamp"], unit="s")
hist = hist.sort_values("datetime")

# Last 12 minutes steady-state window (adjust as needed)
end = hist["datetime"].max()
start = end - timedelta(minutes=12)
steady = hist[(hist["datetime"] >= start) & (hist["datetime"] <= end)]

median_sps = steady["train/steps_per_second"].median()
median_step_time = steady["train/step_time_s"].median()
median_gpu_util = steady["system/gpu_util_percent"].median()

print({
    "median_steps_per_sec": float(median_sps),
    "median_step_time_s": float(median_step_time),
    "median_gpu_util_percent": float(median_gpu_util),
    "samples": int(len(steady)),
})
```

Notes:
- Inspect the run charts first to choose a steady-state period (exclude warm-up)
- You can increase `samples` or use `run.scan_history()` for very long runs
- Consider rolling medians to smooth transient spikes/dips

