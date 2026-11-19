# Sprint 2 completion report

19 November 2026. Implementation complete; researcher review pending. Sprint 3 is not started.

## Implemented

Controlled Gaussian perturbations, IEEE-754 float32 bit flips and structured sign/scale/offset operations, plus healthy controls. Configurable tensor/layer and hidden-neuron targeting; seeded coordinate sampling; copy-only injection; complete attempt/rejection/change records; healthy validation eligibility; validation-only severity assessment; and a separate allowlisted weights-only case export. New CLI commands: `corrupt` and `verify-case`. No new dependencies.

The PRD's three initial fault families are covered; structured tampering has three scalar operations. Targeted class misbehavior and permutation controls remain later extensions. The proposed 85% healthy-validation and 15 percentage-point damage gates are explicit development defaults, not a frozen confirmatory protocol. Exact rules and commands are in `SPRINT_2_PROTOCOL.md`.

## Tests

Full suite: **60 passed in 30.04 seconds**, zero failures/errors/skips, using the existing Windows/Python 3.14.6 CPU environment. After correcting subtraction rounding at severity thresholds, all **26 fault tests passed in 20.54 seconds**. The correction adds 1e-12 absolute tolerance at threshold equality; the existing 34 testbed tests were unaffected.

Tests verify exact repeated-seed edits, complete coordinate/value diffs, source-model preservation, bit XOR arithmetic, NaN/Inf proposal rejection, finite-weight inference overflow retention, targeting, unchanged edits, invalid configs, validation-only evaluation calls, private-data exclusion from exports, checksum tamper detection, ineligible/failed-case retention, and both new CLI commands.

Raw test counts are in `reports/sprint-2/test-results.json`. Case manifests record source/dependency versions and hashes. The smoke cases were generated before the threshold-tolerance fix; their exact historical source is retained under `reports/sprint-2/source_snapshot`. Its bytes match the source hashes in the manifests. All twelve severity labels were checked against final code and are unchanged.

## Actual development results

One fixed seed and six profiles per dataset, using only validation samples: Iris n=15, Wine n=18. All twelve attempted cases completed; all are retained. No test/probe evaluations were performed by this workflow.

| Profile | Iris validation accuracy | Wine validation accuracy |
|---|---:|---:|
| Healthy control | 86.67% | 100.00% |
| Gaussian noise | 93.33% | 100.00% |
| Random bit flips | 100.00% | 100.00% |
| Neuron 0 sign changes | 66.67% | 94.44% |
| Hidden layer zero scaling | 33.33% | 38.89% |
| Uniform output-bias offset | 86.67% | 100.00% |

Three cases meet the development damage threshold: Iris sign changes (20.00 percentage-point drop), Iris hidden-layer scaling (53.33), and Wine hidden-layer scaling (61.11). One has mild degradation (Wine sign, 5.56 points); eight have no loss or improvement. There were zero numerical proposal rejections in these twelve smoke cases; dedicated tests exercise rejected and unscorable outcomes. The apparent improvements on Iris are small-sample development observations and were not selected or tuned for.

The full summary links every case to its damaged checkpoint, private manifest and public export. Original healthy checkpoint hashes were verified unchanged. Negative/no-damage outcomes are included rather than filtered out.

## Limitations and review required

These results demonstrate fault-engine behavior, not LLM detection or repair effectiveness. There is one seed per dataset/profile; input magnitudes are illustrative and are not statistically calibrated severity presets. Larger independent-seed studies still require protocol review. Validation/test arrays exist within the researcher process, but only validation is evaluated. Directory separation is not OS isolation; only `agent/case.json` is eligible for a future agent context. Checksums detect drift relative to a manifest, not an adversary replacing the entire bundle.

Review Sprint 2 interfaces, targeting semantics, thresholds and recorded outcomes. The prior Sprint 1 report remains historical. No main merge is performed. Research licensing remains undecided.

## Next sprint

Sprint 3 requires explicit approval. Its prerequisites are acceptance of the case/export boundary, a decision on weights-only versus bounded diagnostics, diagnosis output schema, provider selection and an API budget. No LLM calls, provider setup or diagnosis functionality were introduced in Sprint 2.
