# Small software demonstration

Generated from the current `run.py` with:

```bash
python run.py --plot --out examples/demo
```

The committed CSV, metadata, and figure are a single seeded 20×20, 10-step
example. They demonstrate runnable software, not a research finding, calibration,
or estimate of real crime. `metadata.json` records the environment and seed.
Step 0 is initialization and is omitted from the figure.

To run without replacing the tracked example, use `python run.py --plot`
(default output: ignored `outputs/demo/`). No full sweep was executed.
