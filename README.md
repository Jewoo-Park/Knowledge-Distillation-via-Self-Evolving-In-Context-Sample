# Confidence-Gated In-Context Samples with a Self-Evolving Memory for Video QA

A frozen small video-language model (Qwen3.5-9B, NF4, 32 frames) answers multiple-choice video questions. When it is
unsure of its answer, it sees one solved example retrieved from a memory of its own past solving records. A teacher
LLM (gpt-6.1-sol) builds and edits that memory after every block of questions. The student's weights never change.

## Method

1. The student answers a block of 50 questions in a fixed order. The memory starts empty. For each question:
    1. It answers without examples (vanilla).
    2. At its answer-letter token, the log-probabilities of the question's own option letters (`>A`, `>B`, ...) are
       converted to probabilities and renormalised. The largest one is the confidence p1.
    3. If p1 ≥ 0.99, the vanilla answer is final.
    4. If p1 < 0.99 or p1 cannot be read, one example is retrieved:
        - the query (video, question, options) is encoded with WeMM-Embedding-2B;
        - candidates are the memory items (active + hold) from other videos;
        - the item whose source question has the highest cosine similarity is inserted into the prompt;
        - the question is answered again.
    5. If the second output is invalid, the vanilla answer is kept.
2. After the block, its labels are revealed and the memory is edited:
    1. **Critic call 1 (select):** from the block's records, choose up to 5 cases worth a visual re-check.
    2. **Teacher call (solve):** solve those cases from frames, question and options only, without the answer.
    3. **Critic call 2 (write):** compare the student's and the teacher's solutions. Write example candidates from
       the cases the teacher solved correctly.
    4. **Critic call 3 (validate, up to 3×):** the student re-answers past questions from other videos with each
       candidate attached. The critic revises the candidates from those results.
    5. **Critic call 4 (manage):** using the memory, the candidates and each item's usage record, apply
       add / merge / prune / refine / hold.
3. The next block is answered with the edited memory, and this repeats to the end.

Teacher and critic are the same model, gpt-6.1-sol. Only the teacher call sees video frames.

## Results (`results_nov/`)

| | Urban Video Bench | VideoSIS | Video-MME |
| --- | --- | --- | --- |
| Questions (blocks) | 5,355 (108) | 4,852 (98) | 2,700 (54) |
| Vanilla → ours | 40.56% → **42.95%** | 57.75% → **59.60%** | 57.15% → **58.11%** |
| Difference [95% CI, video bootstrap] | **+2.39 [+1.40, +3.35]** | **+1.85 [+0.81, +2.93]** | +0.96 [−0.15, +2.07] |
| Fixed / broken | 394 / 266 | 389 / 299 | 136 / 110 |
| Questions given an example | 2,033 (38%) | 2,163 (45%) | 1,019 (38%) |
| Teacher cost | $27.54 | $27.74 | $11.37 |

- `results_nov/results_final/` holds the main experiment:
  - `final_report.md`, `subset_analysis.md`, `protocol.md`;
  - `config_used.json`, the exact config of the reported run;
  - per benchmark: `model_output.jsonl` (one row per question), `memory.jsonl`, `block_curve.jsonl`, `summary.md`.
- `results_nov/results_motiv/` holds the motivation experiment: confidence vs. answer change under one retrieved
  example, on a 20% stratified sample of four benchmarks, with token-level log-probabilities. Low-confidence answers
  (p1 < 0.95) changed 54% of the time and very confident ones (p1 ≥ 0.999) 23% of the time: +31.4 pp [+26.5, +36.2].
- Every JSONL file starts with a summary line.
- The motivation and threshold-selection scripts are not part of this repository. Only their results are.

## Layout

```
configs/final_c.yaml          the method's configuration
scripts/run_icl_confidence.py the run: confidence gate, retrieval, block loop, teacher/critic stages, export
scripts/run_icl_stages.py     the five teacher/critic stages, memory operations, student input
scripts/run_icl_evolution.py  shared setup (data, vanilla answers, query vectors, GPU pool, teacher client) and generation
scripts/final_export.py       per-benchmark results, rewritten during the run
scripts/compute_p1.py         p1 of every vanilla answer (one forward pass per question)
scripts/embed_queries.py      WeMM query vectors (video + question + options)
scripts/prepare_videosis.py   VideoSIS -> sample format
src/                          model adapter (qwen35_vl), datasets, answer parser, prompts, retrieval, memory, teacher client, statistics
tests/                        unit tests (no GPU, no network)
```

## Setup

Requirements: Python 3.10 and an NVIDIA GPU (CUDA 11.8 wheels). The reported run used 4× RTX 2080 Ti (11 GB).

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                        # unit tests, no GPU or network

# weights, at the paths in configs/final_c.yaml (the student's sha256 is checked at startup)
.venv/bin/huggingface-cli download Qwen/Qwen3.5-9B --local-dir models/Qwen/Qwen3.5-9B
.venv/bin/huggingface-cli download tencent/WeMM-Embedding-2B --revision bbd6cd4bf52cfc6716f752a2df80b2706720bd95 \
    --local-dir models/tencent/WeMM-Embedding-2B
.venv/bin/python scripts/configure_api_key.py                        # OpenAI key -> .secrets/openai_api_key
```

`torch`, `transformers` and `bitsandbytes` are pinned exactly. At startup the run compares the student's runtime
(these versions included) with the one that produced the vanilla answers, and stops if they differ.

## Running

```bash
source scripts/icl_env.sh                                            # offline HF, CUDA allocator settings
.venv/bin/python scripts/compute_p1.py --gpus 0,1,2,3                # once: p1 table
.venv/bin/python scripts/run_icl_confidence.py --config configs/final_c.yaml --gpus 0,1,2,3
```

- Benchmarks: download them from their original sources and extract 32 frames per video into `dataset/`:
  - Urban Video Bench: https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench
  - TempCompass: https://huggingface.co/datasets/lmms-lab/TempCompass
  - Video-MME (v1): https://huggingface.co/datasets/lmms-lab/Video-MME
  - VideoSIS: the authors' release (`scripts/prepare_videosis.py` converts its `eval.jsonl`)
- Inputs are expected in `data/inputs/` (not in the repository):
  - `stream.jsonl`: the questions in their fixed order;
  - `vanilla.sqlite` and `videosis_vanilla_rows/`: vanilla answers;
  - `query_embeddings.sqlite` and `encoder_signature.json`: query vectors;
  - `answer_probs.sqlite`: the p1 table.
- Benchmark frames go in `dataset/` and weights in `models/`.
- The OpenAI key is read from `.secrets/openai_api_key`, which is set with `scripts/configure_api_key.py`.
- A run resumes from its last finished block. Teacher responses are cached by request digest.
- `teacher.max_usd` caps the spend.

## Reproducibility

- The student side is deterministic. In a fresh clone and a fresh environment, regenerated vanilla outputs and
  example-run outputs matched the reported run byte for byte (42/42), and so did p1 (12/12).
- The teacher and critic are API models and are not deterministic. A rerun builds a different memory, so its final
  accuracy will differ somewhat from the reported numbers.
- The reported run's per-question records, memories and config are in `results_nov/results_final/`.

## Known issues (reported run)

- In examples, option letters written as plain words (e.g. "D is the only beef option") are not remapped when the
  example's options are shuffled. Only `option D`, `(D)` and `D.` are remapped.
- The stage-4/5 critic prompts say that items are retrieved by their `when_to_use` text and that held items are never
  shown. The run actually retrieved by the source question's vector and also showed held items.
- The 0.99 threshold was chosen on the same questions, using a different (frozen) memory. Results for thresholds
  between 0.95 and 0.995 were within 0.3 pp of each other on average.
