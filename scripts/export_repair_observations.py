"""Export the scored chats and classifier weights used by the portable dashboard."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "reports/repair-pilot"
CASES = (
    ("iris-sign-v2-pro", "Iris", "Sign reversal"),
    ("iris-hidden-layer-v3-pro", "Iris", "Hidden layer erased"),
    ("wine-hidden-layer-v3-pro", "Wine", "Hidden layer erased"),
    ("iris-subtle-054ee00e-v3-pro", "Iris", "Plausible output weights"),
    ("wine-subtle-db6534aa-v3-pro", "Wine", "Plausible output weights"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(study):
    research = study / "Researcher"
    scores = read(research / "trial-results.json")["trials"]
    assignments = {row["filename"]: row for row in read(research / "answer-key.json")["trials"]}
    trials = []
    for score in scores:
        number = score["trial"]
        name = f"trial-{number:02d}"
        entry = assignments[name + ".md"]
        raw = (study / "Trials" / (name + ".md")).read_bytes()
        packet = json.loads(raw.decode("utf-8-sig").strip().splitlines()[-1])
        follow = read(research / "follow-ups" / (name + ".json"))
        if (sha256(raw).hexdigest() != entry["prompt_sha256"]
                or not packet["case_id"] == follow["case_id"] == entry["case_id"]
                or not score["condition"] == follow["ground_truth"] == entry["condition"]
                or score["dataset"] != entry["dataset"]
                or score["training_seed"] != entry["training_config"]["seed"]):
            raise ValueError(f"Mismatched records for {name}")
        trials.append(dict(score, suspicion=follow["suspect_damage"], action=follow["recommended_action"],
                           reason=follow["reason"], output=packet["tensors"]["output.weight"]["values"],
                           hidden=packet["tensors"]["hidden.weight"]["values"],
                           changes=[{"index": a["coordinates"], "healthy": a["before"], "given": a["after"]}
                                    for a in entry["changed_weights"]]))
    pilots = []
    for folder, dataset, fault in CASES:
        path = PILOT / "manual-results" / folder
        result = read(path / "result.json")
        n = result["validation_count"]
        follow_path = path / "follow-up-diagnosis.json"
        test_path = path / "held-out-test.json"
        test = read(test_path) if test_path.is_file() else None
        pilots.append(dict(dataset=dataset, fault=fault, edits=result["edits"],
                           decision=result.get("decision") or "EDIT",
                           validation=dict(n=n, healthy=round(result["healthy"]*n),
                                           damaged=round(result["before"]*n), repaired=result["correct"]),
                           test=dict(n=test["n"], **test["correct"]) if test else None,
                           suspicion=read(follow_path)["diagnosis"] if follow_path.is_file() else None))
    return dict(study=study.name, trials=trials, pilots=pilots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = export(args.study)
    (PILOT / "study-observations.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
