# DECOR: Auditing LLM Deception via Information Manipulation Theory

DECOR audits strategic deception in LLM outputs, grounded in Information Manipulation Theory (IMT; McCornack 1992). Given a context and a model's response, DECOR:

1. **Decomposes** the context into atomic **Information Units (IUs)** and assigns each a strategic impact weight (1–3).
2. **Audits** how the response handles each IU across four IMT dimensions — **Quantity, Quality, Relation, Manner**.
3. **Aggregates** the per-IU, per-dimension scores into a single deception index M(R).

This repo has the final DECOR pipeline (single-turn + multi-turn), its evaluation code, and the black-box baselines it's compared against — no ablations, no exploratory variants.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # then set OPENAI_API_KEY=...
```

Run DECOR on the single-turn dataset with GPT-4o and evaluate it:

```bash
python workflows/singleturn_iu_audit_workflow.py --model openai_gpt4o
python scripts/eval_imt.py --all
```

That's it — results land in `results/eval/imt/summary.txt` (AUROC, F1, and friends). Swap `--model openai_gpt4o` for any other key in `config.LLM_PRESETS` to use a different model/provider.

---

## Repository Structure

```
agents/         DECOR's agents — InformationUnitAnalyst (Phase 1), IMTAuditor (Phase 2), MultiTurnIUAuditor
workflows/      Pipeline entry points (run the agents over a dataset)
baselines/      5 black-box baselines DECOR is compared against, one folder each
scripts/        Evaluation entry points (AUROC/F1 etc.) + the OpenDeception parser
llm_interface/  LLM backends — OpenAI-protocol / Anthropic / Google, picked via config.py
data/           Datasets: data/singleturn/ (745 labeled responses) and data/open_deception/
config.py       Model presets — this is where --model values come from
```

---

## Running DECOR

### Single-turn

```bash
python workflows/singleturn_iu_audit_workflow.py --model openai_gpt4o
python scripts/eval_imt.py --all
```

### Multi-turn (OpenDeception dialogues)

The default English dataset contains 146 dialogues across GPT-3.5 (45), GPT-4o (47), Llama-3.1-70B (40), and Llama-3.1-8B (14). The parser reads `data/open_deception/OpenDeception-C187/English/` and excludes 10 `fail` and 44 `rejection` transcripts from the 200 source files. It writes `data/open_deception/open_deception_en.json`, shared by DECOR and all multi-turn baseline runners. The older `transcripts/` directory contains only the GPT-4o subset and is no longer the default input.

```bash
python scripts/parse_open_deception.py                        # one-time: build the dialogue dataset
python workflows/multiturn_iu_sis_workflow.py --model openai_gpt4o
python scripts/eval_multiturn_open_deception.py --target response
python scripts/eval_multiturn_open_deception.py --target thought
python scripts/eval_multiturn_open_deception.py --target combined   # response-as-gt: mean(thought, response)
```

`--target combined` scores the post-hoc mean of the independently-computed thought and response dialogue scores — no extra LLM call, no joint-audit logic — against OpenDeception's existing dialogue label (which is already an outcome/response-level label, so `--target thought` and `--target response` above already share the same ground truth). For a baseline whose round file bundles both sides in one JSON (DeceptionBench, CoT Red-Handed, Constitutional Monitor), just pass that one `--round-path`; for one that outputs a separate file per side (Zero-shot/Few-shot via `run_simple_llm_judge_open_deception_round.py`), also pass `--round-path-thought <thought-side file>` alongside `--round-path <response-side file>`.

### Baselines

```bash
python baselines/simple_llm_judge/run_simple_llm_judge.py --model openai_gpt4o            # Zero-shot
python baselines/simple_llm_judge/run_simple_llm_judge_fewshot.py --model openai_gpt4o    # Few-shot
python baselines/deceptionbench/run_deceptionbench.py --model openai_gpt4o                # DeceptionBench
python baselines/cot_red_handed/run_cot_red_handed.py --model openai_gpt4o                # CoT Red-Handed
python baselines/scheming_judge/run_scheming_judge.py --model openai_gpt4o                # Constitutional Monitor

python scripts/eval_baselines.py --list         # see available families
python scripts/eval_baselines.py --family simple_llm_judge_zeroshot --all
```

(Every command's `--model` defaults to `openai_gpt4o`; the explicit flag above is just for clarity.)

Each baseline also has a `_open_deception_round.py` twin for the multi-turn setting (same `--model` flag), evaluated the same way via `scripts/eval_multiturn_open_deception.py --round-path <file>`.

### Human-annotated evaluation set

`data/singleturn/` is a single turn human-annotated set of 745 labeled responses:

```bash
python scripts/run_imt.py --dataset human_eval --model openai_gpt4o

# Or run every preset in config.LLM_PRESETS as an auditor (needs all of their keys set in .env):
python scripts/run_imt.py --dataset human_eval --all
```

---

## Evaluation design: response-as-ground-truth

The single-turn evaluators (`scripts/eval_imt.py`, `scripts/eval_baselines.py`) report two conditions against the **Response's own human label**.

- **Response**: the response-only score.
- **Thought & Response** (`combined`): the arithmetic mean `(thought_score + response_score) / 2`.

Combined reuses the two existing monitor scores without an extra API call. CoT Red-Handed's combined evaluation uses their arithmetic mean; its stored 0.55/0.45 hybrid score is not used here. Both DECOR sides use SIS-weighted means over all IUs. AUROC/AUPRC preserve continuous scores. JSON and summaries contain only these two headline conditions; DECOR's own-label OOF annotations remain diagnostic outputs.

Binary metrics retain method-specific threshold conventions: DECOR selects per-domain thresholds on training folds of five-fold CV, binary judges use their emitted labels (combined ties at 0.5 are positive), CoT Red-Handed uses raw scores >= 50, and Constitutional Monitor uses scores > 5. These F1/accuracy values do not share a threshold-selection protocol; AUROC is the primary comparison metric.

The multi-turn evaluator (`scripts/eval_multiturn_open_deception.py`) follows the same idea via `--target combined` (see above) — OpenDeception's dialogue label is already response/outcome-level, so Thought and Response there already share one ground truth; Combined just adds the post-hoc mean of the two dialogue-level scores.

---

## Using a different model

`--model` accepts any key from `config.LLM_PRESETS`. By default this README uses `openai_gpt4o` (official OpenAI API — just needs `OPENAI_API_KEY`); `anthropic_claude_sonnet46` and `google_gemini25pro` work the same way with their own official key. To add a model that isn't listed, add an entry to `config.py` — see `llm_interface/` for what each provider needs.

---

## License

Code is [MIT](LICENSE). `data/open_deception/OpenDeception-C187/` is third-party data (Wu et al., 2025) redistributed under its own terms — not covered by the MIT license.
