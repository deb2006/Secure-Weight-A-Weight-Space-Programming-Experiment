# Sprint 1 recorded evidence

These are actual development baseline runs, not LLM-repair findings. See [the completion report](../../docs/SPRINT_1_REPORT.md) for executed tests, limitations and review decisions.

- `iris-seed42-sprint1/`: primary Iris run, 20/22 correct test predictions.
- `iris-seed42-rerun/`: same configuration rerun to check exact determinism, not an independent replicate.
- `wine-seed42-sprint1/`: Wine run, 27/27 correct test predictions.
- `reproducibility.json`: exact comparisons across the two Iris runs, including saved checkpoint bytes.
- `test-results.json`: final automated test outcome and retained earlier test-setup failures.

All run files were copied byte-for-byte from local outputs. No metrics were manually edited. Each run manifest hashes its artifact files; the manifest itself is not signed. Source-file hashes match implementation commit `ce4af5c`. The runs predate the first commit and retain null Git revisions in their original provenance.

Verify after installation:

```powershell
.\.venv\Scripts\python.exe -m secureweight verify --run reports/sprint-1/iris-seed42-sprint1
.\.venv\Scripts\python.exe -m secureweight verify --run reports/sprint-1/iris-seed42-rerun
.\.venv\Scripts\python.exe -m secureweight verify --run reports/sprint-1/wine-seed42-sprint1
```

`healthy.pt` contains CPU tensors, architecture and training-only preprocessing. `weights.json` is the corresponding structured, full-precision float32 representation. These are researcher reference artifacts, not inputs for a future repair agent. Preserve their original versions.
