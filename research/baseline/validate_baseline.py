"""Validate the packaged LLM2Rec compatibility baseline contract."""

from __future__ import annotations

import json
from pathlib import Path


BASELINE = Path(__file__).resolve().parent
CONFIG = BASELINE / "config" / "compatibility-profile.json"
REQUIRED_SCRIPTS = {
    "csft",
    "iem",
    "evaluation",
    "audit",
}


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stages = config["stages"]
    stage_ids = {stage["id"] for stage in stages}
    if stage_ids != REQUIRED_SCRIPTS:
        raise RuntimeError(f"stage contract mismatch: {stage_ids}")
    if config["verdict"] != "compatibility-profile":
        raise RuntimeError("baseline verdict must remain compatibility-profile")
    if config["source"]["commit"] != "73b481f710f67166ab958f4985d27b27fb410871":
        raise RuntimeError("upstream source commit drifted")
    if config["compute"]["accelerator"] != "NvidiaTeslaT4":
        raise RuntimeError("compute profile drifted from the verified T4 baseline")
    stage_map = {stage["id"]: stage for stage in stages}
    csft = stage_map["csft"]["config"]
    if csft["max_steps"] != 1000 or csft["num_epochs"] != 1:
        raise RuntimeError("CSFT budget contract drifted")
    if stage_map["iem"]["outputs"] != ["checkpoint-500", "checkpoint-1000"]:
        raise RuntimeError("IEM checkpoint contract drifted")
    evaluation = stage_map["evaluation"]["config"]
    if evaluation["checkpoints"] != [500, 1000] or evaluation["seeds"] != [2024, 2025, 2026]:
        raise RuntimeError("evaluation checkpoint or seed contract drifted")
    audit = stage_map["audit"]["config"]
    if audit["raw_rank_tolerance"] != 1e-8 or audit["required_users"] != 15323:
        raise RuntimeError("audit contract drifted")
    required_files = [
        BASELINE / stage["script"] for stage in stages
    ] + [
        BASELINE / "provenance" / "upstream-contract.json",
        BASELINE / "provenance" / "reproduction-provenance.json",
    ]
    missing = [str(path.relative_to(BASELINE)) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"baseline files missing: {missing}")
    print(json.dumps({"baseline": config["name"], "stages": [stage["id"] for stage in stages]}, indent=2))


if __name__ == "__main__":
    main()
