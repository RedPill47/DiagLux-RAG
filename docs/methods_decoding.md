# Methods: models, decoding, and the heterogeneity caveat

Draft text and configuration record for the paper's experimental-setup section.
Effective per-model settings are validated live (2026-06-13) and recorded in every
run's `preds_*.config.json` sidecar. The variance figures below are filled from the
control runs.

## Models

We evaluate four contemporary instruction-tuned LLMs spanning two providers and a
deliberate capability range, rather than a single family, to characterize whether
*existing* LLMs can read Luxembourgish (RQ1):

| Model | Provider | Reasoning | Decoding (effective) | Output budget |
|---|---|---|---|---|
| GPT-5.5 | OpenAI | yes (always) | temperature = model default (1); `temperature=0` rejected | `max_completion_tokens` 8192 |
| deepseek-v4-pro | DeepSeek | yes (always) | temperature = 0 (deterministic requested) | `max_tokens` 8192 |
| Claude Opus 4.8 | Anthropic | no (direct) | temperature = model default (parameter deprecated) | `max_tokens` 64 |
| Claude Sonnet 4.6 | Anthropic | no (direct) | temperature = 0 (deterministic requested) | `max_tokens` 2048 |

All four receive the identical strict prompt (one Luxembourgish MC question, four
options, "return only the letter"), with answer options shuffled per question under a
fixed seed and the model's chosen letter mapped back to its semantic option type.
Outputs are parsed by a robust letter parser; unparseable outputs are logged as a
separate category, never silently scored wrong. (Sonnet 4.6 was given a larger 2048-token
budget than the other Claude model because it tends to reason aloud before answering;
64 tokens truncated it before the letter, so the budget was raised and the parser made
robust to chain-of-thought.)

## Decoding is necessarily heterogeneous

These models cannot be placed in a single, uniform decoding regime, and we do not
force one:

- **Temperature.** GPT-5.5 only supports its default temperature and rejects 0;
  Claude Opus 4.8 has deprecated the temperature parameter entirely. Only
  deepseek-v4-pro and Claude Sonnet 4.6 accept `temperature=0`. We therefore run each
  model in the most deterministic configuration it permits and record the *effective*
  temperature per run.
- **Reasoning.** GPT-5.5 and deepseek-v4-pro are reasoning models that emit hidden
  reasoning tokens before answering (roughly 400 to 1000 tokens for a single MC item),
  which is why they require a large completion budget; the Claude models answer directly.
  The large budgets are ceilings only; billing is on tokens actually generated.
- **Token parameter.** GPT-5.5 requires `max_completion_tokens`; the others use
  `max_tokens`. The harness adapts to these API constraints automatically and records
  what it used.

We argue this heterogeneity does not undermine the study, for three reasons. First, a
fully controlled cross-model comparison is unattainable regardless: the models differ
in training data, tokenizer, architecture, and alignment, so equalizing temperature
would remove two visible confounds while leaving many. Second, every **load-bearing
comparison in this paper is within a single model**: closed-book vs. full-text oracle
vs. RAG@k, accuracy by cognitive type, and the retrieval-failure vs. comprehension-failure
decomposition, and these hold decoding fixed by construction. The cross-model
ranking is reported descriptively, not as a controlled claim. Third, reasoning models
*are* the current frontier; running them in an artificial non-reasoning mode (where
even possible) would reduce external validity for the question "can existing LLMs read
Luxembourgish." We instead treat test-time reasoning as an analysis dimension: two
models reason, two answer directly, and we report whether that distinction tracks
accuracy on this low-resource comprehension task.

## Run-to-run variance check

Because GPT-5.5 and Claude Opus 4.8 run at a non-zero, uncontrolled temperature, their
accuracies carry sampling noise. To ensure cross-model gaps are interpreted against the
noise floor, we run both models **three times** on a fixed seeded subset of 64
questions (4 per text across 16 texts; `scripts/make_variance_sample.py`, seed 13) in the
full-text oracle setting, and report the mean and range. deepseek-v4-pro and
Sonnet 4.6 (temperature 0) are nominally deterministic and are not repeated.

Observed oracle accuracy across the three repeats:

| Model | rep 1 | rep 2 | rep 3 | mean | range |
|---|---|---|---|---|---|
| GPT-5.5 | 0.797 | 0.844 | 0.786 (partial) | 0.809 | at least 0.047 |
| Claude Opus 4.8 | 0.859 | 0.859 | 0.859 | 0.859 | **0.000** |

(GPT-5.5 rep 3 was truncated at 28/64 questions when the OpenAI account hit its quota;
reported for context only. The two complete repeats already span 4.7 points.)

The two non-deterministic models behave very differently. **Claude Opus 4.8** returned
*identical* accuracy across all three repeats, effectively deterministic on this MC task
despite its uncontrolled temperature. **GPT-5.5**, forced to temperature 1, shows a real
spread of about 5 points. Interpretation rule: cross-model gaps **involving GPT-5.5**
smaller than about 5 points are reported as ties; Opus (and the temperature-0 models,
deepseek-v4-pro and Sonnet 4.6) carry no meaningful decoding noise. This is exactly why
the check was run: the noise floor is model-specific, and only GPT-5.5 needs the caution.

## Reproduction

```
# variance sample (committed): outputs/processed/questions_variance_sample.jsonl
python scripts/make_variance_sample.py

# three oracle repeats per non-deterministic model, separate out-dirs
for rep in 1 2 3; do
  python scripts/run_answering.py --system oracle --provider openai --model gpt-5.5 \
    --max-tokens 8192 --questions outputs/processed/questions_variance_sample.jsonl \
    --out-dir outputs/runs/variance/rep$rep
  python scripts/run_answering.py --system oracle --provider anthropic --model claude-opus-4-8 \
    --questions outputs/processed/questions_variance_sample.jsonl \
    --out-dir outputs/runs/variance/rep$rep
done
```
