# Sprint 2 development evidence

`summary.json` lists all twelve development cases. Each opaque case directory has a researcher manifest/checkpoint, a separate weights-only `agent/case.json`, and a checksum index. Never provide the whole case directory, summary, source snapshots or researcher records to a diagnosis agent.

`test-results.json` records the 60-test full-suite outcome. The subsequent threshold-boundary correction passed all 26 fault tests. `source_snapshot/secureweight` preserves the exact source used for these smoke cases, verified against their manifests; it predates only the numerical threshold-tolerance correction. `provenance-check.json` records source correspondence and checks that final code gives identical severity labels for all saved results.

See `../../docs/SPRINT_2_REPORT.md` and `../../docs/SPRINT_2_PROTOCOL.md` for interpretation and commands. Run `python -m secureweight verify-case --case <case-directory>` after installation to verify hashes and public-export/checkpoint correspondence. Evidence is saved without Git newline conversion.
