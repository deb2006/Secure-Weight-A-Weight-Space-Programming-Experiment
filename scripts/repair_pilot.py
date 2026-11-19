"""Build a portable pilot dashboard or serve it locally with first-response scoring."""
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import secrets

import torch

from secureweight.config import load_config
from secureweight.data import prepare_data
from secureweight.faults.cases import verify_case
from secureweight.models.checkpoints import load_checkpoint
from secureweight.tracking import file_hash, object_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "reports" / "repair-pilot"
BUDGET = 4
TEMPLATE = ROOT / "scripts" / "repair_dashboard.html"
INSTRUCTION = """Review this small classifier using only the supplied information. It may be healthy.
Propose up to four scalar replacements, or return an empty edits array if you would leave it alone.
Use zero-based tensor indices. Reply once, using only a JSON object of the form
{"case_id":"<given ID>","edits":[{"tensor":"<name>","index":[0,0],"value":0.0}]}.
Do not use tools, run inference, browse, retrain, or ask for examples."""


def build_cases():
    cases = {}
    for path in sorted((ROOT / "reports/sprint-2/cases").iterdir()):
        manifest = verify_case(path)
        run = ROOT / "reports/sprint-1" / manifest["source_run_id"]
        meta = read_json(run / "dataset.json")
        agent = read_json(path / "agent/case.json")
        weights = agent["weights"]
        dataset = meta["dataset"]
        task = {"iris": "Predict the Iris species from four flower measurements.",
                "wine": "Predict the wine cultivar from thirteen chemical measurements."}[dataset]
        packet = dict(case_id=agent["case_id"], task=task, features=meta["feature_names"],
                      classes_in_output_order=agent["class_names"],
                      standardization={k:agent["preprocessing"][k] for k in ("mean","scale")},
                      forward="x=(raw-mean)/scale; h=relu(hidden.weight @ x + hidden.bias); predict argmax(output.weight @ h + output.bias). Rows index output units.",
                      architecture=weights["architecture"],
                      tensors={k: {p:v[p] for p in ("shape","dtype","values")}
                               for k,v in weights["tensors"].items()})
        prompt = INSTRUCTION + "\n\n" + json.dumps(packet, allow_nan=False)
        cid = agent["case_id"]
        cases[cid] = dict(id=cid, dataset=dataset, packet=packet, prompt=prompt,
                          original=manifest["healthy_validation"]["accuracy"],
                          before=(manifest["damaged_validation"] or {}).get("accuracy"),
                          n=len(manifest["validation_indices"]), path=path, run=run, manifest=manifest)
    return cases


def apply_response(model, cid, response):
    if not isinstance(response, dict) or set(response) != {"case_id","edits"} or response["case_id"] != cid:
        raise ValueError("Expected case_id and edits for the selected case.")
    edits = response["edits"]
    if not isinstance(edits, list) or len(edits) > BUDGET:
        raise ValueError("Expected a list with at most four edits.")
    seen = set()
    tensors = dict(model.named_parameters())
    with torch.no_grad():
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"tensor","index","value"}:
                raise ValueError("Each edit requires tensor, index and value.")
            name, index, value = edit["tensor"], edit["index"], edit["value"]
            tensor = tensors.get(name) if isinstance(name,str) else None
            if tensor is None or not isinstance(index,list) or len(index) != tensor.ndim:
                raise ValueError("Unknown tensor or incorrect index length.")
            if any(type(i) is not int or not 0 <= i < size for i,size in zip(index,tensor.shape)):
                raise ValueError("Index outside tensor.")
            key = (name,tuple(index))
            if key in seen or type(value) not in (int,float) or not math.isfinite(value) or abs(value)>torch.finfo(torch.float32).max:
                raise ValueError("Repeated coordinate or invalid float32 replacement.")
            seen.add(key)
            tensor[tuple(index)] = value
    return len(seen)


def score(case, raw, model_name):
    cid, run, manifest = case["id"], case["run"], case["manifest"]
    result = dict(case_id=cid, model=model_name, raw_response=raw,
                  time_utc=datetime.now(timezone.utc).isoformat(), prompt_hash=object_hash(case["prompt"]),
                  damaged_checkpoint=manifest["damaged_model_hash"], partition="validation")
    if raw is None:
        return dict(result, status="missing", reason="Chat showed completion without a recoverable reply.")
    damaged, _ = load_checkpoint(case["path"]/"researcher/damaged.pt", manifest["damaged_model_hash"])
    healthy, _ = load_checkpoint(run/"healthy.pt", manifest["source_model_hash"])
    data = prepare_data(load_config(run/"config.json"))
    if data.metadata != read_json(run/"dataset.json"):
        raise ValueError("Saved data split does not match.")
    candidate = deepcopy(damaged)
    try:
        result["edits"] = apply_response(candidate, cid, json.loads(raw))
    except (ValueError, OverflowError, RecursionError) as exc:
        return dict(result, status="invalid", reason=str(exc))
    with torch.inference_mode():
        logits = [m(data.validation.x) for m in (healthy, damaged, candidate)]
    if any(not torch.isfinite(values).all() for values in logits):
        return dict(result,status="unscorable")
    clean, before, after = [v.argmax(1) for v in logits]
    preserved = clean == before
    correct = int((after == data.validation.y).sum())
    return dict(result,status="scored",correct=correct,after=correct/len(after),
                preserved=int(((after==clean)&preserved).sum()), unaffected=int(preserved.sum()))


def completed_repairs():
    results = []
    for dataset in ("Iris", "Wine"):
        folder = PILOT / "manual-results" / f"{dataset.lower()}-hidden-layer-v3-pro"
        validation_path = folder / "result.json"
        validation = read_json(validation_path)
        test = read_json(folder / "held-out-test.json")
        if (test["source_trial_sha256"] != file_hash(validation_path)
                or test["case_id"] != validation["case_id"]
                or test["normalized_patch"] != validation["normalized_patch"]):
            raise ValueError(f"{dataset} scores do not refer to the same saved patch")
        n = validation["validation_count"]
        results.append({
            "dataset": dataset, "case": test["case_id"],
            "edits": validation["normalized_patch"]["edits"],
            "validation": {"n": n, "healthy": round(validation["healthy"] * n),
                           "damaged": round(validation["before"] * n), "repaired": validation["correct"]},
            "test": {"n": test["n"], **test["correct"]}, "paired": test["paired"],
        })
    return results


def render(cases, token=""):
    rows = []
    for case in cases.values():
        row = {k:v for k,v in case.items() if k not in ("manifest","run","path")}
        response = PILOT/"responses"/(row["id"]+".json")
        row["result"] = read_json(response) if response.is_file() else None
        rows.append(row)
    payload = dict(rows=rows,token=token,repairs=completed_repairs(),
                   study=read_json(PILOT / "study-observations.json"))
    page = TEMPLATE.read_text(encoding="utf-8")
    for marker, name in (("__STUDY_CSS__", "repair_study.css"), ("__STUDY_JS__", "repair_study.js")):
        page = page.replace(marker, (TEMPLATE.parent / name).read_text(encoding="utf-8"))
    paper = base64.b64encode((PILOT / "SecureWeight-IEEE.pdf").read_bytes()).decode("ascii")
    page = page.replace("__PAPER_BASE64__", paper)
    return page.replace("__PILOT_DATA__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))


def serve(cases, port):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Frame-Options","DENY")
            self.end_headers()
            self.wfile.write(render(cases,token).encode())

        def do_POST(self):
            if self.path != "/response" or self.headers.get("X-Pilot-Token") != token:
                return self.send_error(403)
            try:
                size = int(self.headers.get("Content-Length","0"))
                if not 0<size<=64000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                case = cases[body["case_id"]]
                raw, model = body["response"], body["model"]
                if not isinstance(raw,str) or not isinstance(model,str) or not model.strip() or not raw.strip():
                    raise ValueError("Enter the model name and its complete first response.")
                result = score(case,raw,model.strip())
                out = PILOT/"responses"
                out.mkdir(exist_ok=True)
                write_json(out/(case["id"]+".json"), result)
            except FileExistsError:
                return self.send_error(409,"Case already recorded")
            except (ValueError, KeyError, TypeError):
                return self.send_error(400,"Invalid submission")
            self.send_response(201)
            self.end_headers()
    with HTTPServer(("127.0.0.1",port),Handler) as server:
        print(f"Pilot dashboard http://127.0.0.1:{port}/",flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--port",type=int,default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = build_cases()
    PILOT.mkdir(parents=True,exist_ok=True)
    (PILOT/"index.html").write_text(render(cases),encoding="utf-8")
    print(f"Prepared {len(cases)} verified cases in {PILOT}",flush=True)
    if args.serve:
        serve(cases,args.port)
