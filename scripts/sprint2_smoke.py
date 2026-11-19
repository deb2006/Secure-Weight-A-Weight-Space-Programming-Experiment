"""Exercise every development profile once per baseline, retaining every case."""
from pathlib import Path
from secureweight.faults import load_fault_config
from secureweight.faults.cases import create_case, verify_case
from secureweight.tracking import write_json


def main():
    output = Path("reports/sprint-2")
    if (output / "summary.json").exists():
        raise FileExistsError("Evidence bundle already exists")
    rows = []
    for dataset in ("iris", "wine"):
        source = Path("reports/sprint-1") / f"{dataset}-seed42-sprint1"
        for profile in ("healthy", "gaussian", "bit-flip", "sign", "scale", "offset"):
            case = create_case(source, load_fault_config(f"configs/faults/{profile}.yaml"), output / "cases")
            m = verify_case(case)
            row = {"dataset": dataset, "profile": profile, "case": case.relative_to(output).as_posix(),
                   "status": m["status"], "attempted": m.get("attempted_count"),
                   "changed": m.get("changed_count"), "rejected": m.get("rejected_count"),
                   "healthy_accuracy": m["healthy_validation"]["accuracy"],
                   "damaged_accuracy": (m.get("damaged_validation") or {}).get("accuracy"),
                   "severity": m.get("severity")}
            rows.append(row)
            print(dataset, profile, row["severity"], flush=True)
    write_json(output / "summary.json", {"protocol": "sprint-2-development-v1", "cases": rows,
        "interpretation": "One seed per dataset/profile; validation-only engineering smoke cases, not calibrated study bands or repair results."})


if __name__ == "__main__":
    main()
