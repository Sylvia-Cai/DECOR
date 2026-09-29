"""Parse OpenDeception transcript files into structured JSON.

Usage:
    python scripts/parse_open_deception.py --input \
        "data/open_deception/OpenDeception-C187/transcripts/cheat_success/Fake Recruitment Assistant Privacy9.txt"

    # Regenerate the curated dataset every other script/workflow reads by default
    # (data/open_deception/open_deception_en.json — not checked in, since it
    # is a deterministic function of the raw transcripts above):
    python scripts/parse_open_deception.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import (  # noqa: E402
    OPEN_DECEPTION_EXAMPLES as OPENDECEPTION_EXAMPLES,
    OPEN_DECEPTION_INPUT as DEFAULT_OUTPUT,
    OPEN_DECEPTION_RAW_DIR as DEFAULT_INPUT,
)

LABEL_NAMES = {"cheat_fail", "cheat_none", "cheat_success", "fail", "rejection"}
EVALUATION_LABELS = {"cheat_fail", "cheat_none", "cheat_success"}


def _normalize_scenario_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\bfraud\d+\b", "fraud", s)
    s = re.sub(r"\bsalesman\d+\b", "salesman", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _lookup_goal(goal_map: dict[str, str], scenario_name: str) -> str | None:
    goal = goal_map.get(scenario_name)
    if goal:
        return goal
    norm_target = _normalize_scenario_name(scenario_name)
    for k, v in goal_map.items():
        if _normalize_scenario_name(k) == norm_target:
            return v
    return None


def _load_open_deception_goals() -> dict[str, str]:
    if not OPENDECEPTION_EXAMPLES.exists():
        return {}
    text = OPENDECEPTION_EXAMPLES.read_text(encoding="utf-8")
    pattern = re.compile(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]')
    goals: dict[str, str] = {}
    for m in pattern.finditer(text):
        ai_role, goal, _user_role, _start_msg = m.groups()
        goals[ai_role] = goal
    return goals


def _extract_dialogue_label(path: Path) -> str:
    for part in path.parts:
        if part in LABEL_NAMES:
            return part
    return "unknown"


def parse_transcript(path: Path, goal_map: dict[str, str]) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    rounds = []
    current_user = None
    current_thought = None
    current_speech = None
    current_field = None
    round_id = 0

    def flush_round() -> None:
        nonlocal current_user, current_thought, current_speech, round_id
        if current_user is None and current_thought is None and current_speech is None:
            return
        if current_user is not None and current_thought is not None and current_speech is not None:
            round_id += 1
            rounds.append(
                {
                    "round_id": round_id,
                    "user": current_user,
                    "ai_thought": current_thought,
                    "ai_speech": current_speech,
                }
            )
        current_user = None
        current_thought = None
        current_speech = None

    for line in lines:
        if line == "<END>":
            continue

        if line.startswith("user:"):
            flush_round()
            current_user = line[len("user:") :].strip()
            current_field = "user"
            continue

        if line.startswith("[AI (Thought):"):
            current_thought = line[len("[AI (Thought):") :].strip()
            if current_thought.endswith("]"):
                current_thought = current_thought[:-1].rstrip()
            current_field = "thought"
            continue

        if line.startswith("AI (Speech):"):
            current_speech = line[len("AI (Speech):") :].strip()
            current_field = "speech"
            continue

        # Merge unexpected continuation lines into the latest open field.
        if current_field == "speech":
            current_speech = f"{current_speech}\n{line}"
        elif current_field == "thought":
            if line.endswith("]"):
                line = line[:-1].rstrip()
            current_thought = f"{current_thought}\n{line}"
        elif current_field == "user":
            current_user = f"{current_user}\n{line}"

    flush_round()

    scenario_name = path.stem
    deceptive_goal = _lookup_goal(goal_map, scenario_name)

    try:
        source_file = str(path.relative_to(ROOT))
    except ValueError:
        source_file = str(path)
    out = {
        "source_file": source_file,
        "dialogue_label": _extract_dialogue_label(path),
        "rounds": rounds,
    }
    if deceptive_goal:
        out["goal"] = deceptive_goal
    return out


def collect_transcripts(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse OpenDeception transcripts into structured JSON")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input transcript file or directory")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Also parse fail/rejection transcripts (excluded by default)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    transcripts = collect_transcripts(input_path)
    if not args.include_excluded:
        transcripts = [p for p in transcripts if _extract_dialogue_label(p) in EVALUATION_LABELS]
    if not transcripts:
        parser.error(f"No eligible transcripts found in {input_path}")
    goal_map = _load_open_deception_goals()
    parsed = [parse_transcript(path, goal_map) for path in transcripts]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(parsed if len(parsed) > 1 else parsed[0], f, ensure_ascii=False, indent=2)

    print(f"Parsed {len(parsed)} transcript(s) -> {output_path}")


if __name__ == "__main__":
    main()
