# Final Results: Urban Video Bench, VideoSIS, Video-MME (2026-11-02)

We compare confidence-gated in-context samples with a self-evolving memory (ours) against vanilla. Both conditions
answer the same questions in the same order, and every comparison below is paired per question.
- The protocol is in `protocol.md`.
- Per-benchmark details are in `urban/`, `videosis/` and `videomme/`, each with a `summary.md`.
- The exact config is in `config_used.json`.

## Method summary
- **Student:** Qwen3.5-9B (NF4, 32 frames, greedy decoding). Its weights are never updated.
- **When to give an example.**
  - At the vanilla answer-letter token (`>X`), we take the probabilities of the question's own option letters and
    renormalise them over the options. The largest value is p1.
  - If p1 < 0.99, or p1 cannot be read, the question gets an example.
  - Otherwise the vanilla answer is final, and the input is byte-identical to the vanilla run.
- **Retrieval.**
  - From the benchmark's memory (active + hold items), we take the one item whose source question has the highest
    WeMM-embedding cosine similarity with the current question.
  - Items from the same video are excluded.
  - There is no similarity floor.
  - If the output with the example is invalid, the vanilla answer is kept.
- **Memory.**
  - The memory starts empty.
  - After each block of 50 questions, the labels are revealed, and gpt-6.1-sol (teacher and critic) edits the memory
    in five stages: select, solve blind, write candidates, validate with student trials, and manage
    (add / merge / prune / refine / hold).
  - Example texts are limited to 1,300 characters.

## 1. Overall results

| | Urban Video Bench | VideoSIS | Video-MME |
|---|---|---|---|
| Questions (blocks) | 5,355 (108) | 4,852 (98) | 2,700 (54) |
| Vanilla accuracy | 40.56% | 57.75% | 57.15% |
| **Our accuracy** | **42.95%** | **59.60%** | **58.11%** |
| **Difference** | **+2.39 pp** | **+1.85 pp** | **+0.96 pp** |
| 95% CI (video-cluster bootstrap) | [+1.40, +3.35] | [+0.81, +2.93] | [−0.15, +2.07] |
| Fixed / broken | 394 / 266 | 389 / 299 | 136 / 110 |
| Sign test p | 3.6 × 10⁻⁷ | 3.4 × 10⁻⁴ | 0.055 |
| Teacher (sol) cost | $27.54 (710 calls) | $27.74 (670 calls) | $11.37 (357 calls) |

- Urban and VideoSIS improve significantly.
- Video-MME improves, but the gain falls just short of significance.

## 2. Questions with and without an example

| | Urban | VideoSIS | Video-MME |
|---|---|---|---|
| Gated (p1 < 0.99 or unreadable) | 2,059 (38.5%) | 2,187 (45.1%) | 1,052 (39.0%) |
| Given an example | 2,033 (38.0%) | 2,163 (44.6%) | 1,019 (37.7%) |
| Gated, but no item available (early blocks) | 26 | 24 | 33 |
| Not gated (confident) | 3,296 | 2,665 | 1,648 |
| **Accuracy on questions given an example (vanilla → ours)** | **32.2% → 38.5% (+6.30 pp)** [+3.62, +8.81] | **51.8% → 55.9% (+4.16 pp)** [+1.86, +6.58] | **42.0% → 44.6% (+2.55 pp)** [−0.59, +5.52] |
| Answer changed among them | 58% (1,171) | 44% (945) | 41% (419) |

- Questions without an example have the same input as vanilla, so their answers and accuracy are identical
  (difference 0).

## 3. Questions given an example, split by vanilla confidence

| | Urban | VideoSIS | Video-MME |
|---|---|---|---|
| Low confidence (p1 < 0.99) | 1,892 q, **+4.86 pp** [+2.25, +7.48] (+349 / −257) | 2,111 q, **+4.17 pp** [+1.70, +6.55] (+374 / −286) | 938 q, +0.64 pp [−2.56, +3.75] (+113 / −107) |
| p1 unreadable (vanilla output mostly broken) | 141 q, **+25.5 pp** [+16.4, +34.9] (+45 / −9) | 52 q, +3.85 pp [−16.0, +23.6] (+15 / −13) | 81 q, **+24.7 pp** [+14.1, +36.3] (+23 / −3) |

- On Urban and VideoSIS, examples help significantly even on questions whose confidence could be read normally.
- On Video-MME, most of the gain comes from questions whose vanilla output was broken: it looped or was cut off at
  the token limit. With an example, the student answered in the proper format.

**Difference (pp) by p1 band, on questions given an example** (number of questions in parentheses)

| p1 band | < 0.5 | 0.5–0.9 | 0.9–0.95 | 0.95–0.97 | 0.97–0.98 | 0.98–0.99 | unreadable |
|---|---|---|---|---|---|---|---|
| Urban | +6.8 (145) | +8.3 (653) | −0.8 (245) | −2.7 (222) | +10.6 (196) | +3.2 (431) | +25.2 (141) |
| VideoSIS | 0.0 (44) | +6.4 (586) | +1.2 (345) | +6.5 (285) | +8.5 (281) | +0.5 (570) | +3.8 (52) |
| Video-MME | +2.7 (72) | +0.7 (292) | +6.0 (129) | −3.7 (132) | −2.9 (100) | +0.9 (213) | +24.4 (81) |

## 4. Change over the stream (all questions, blocks split into thirds)

| Third | Urban (blocks 1–36 / 37–72 / 73–108) | VideoSIS (1–33 / 34–66 / 67–98) | Video-MME (1–18 / 19–36 / 37–54) |
|---|---|---|---|
| First | +2.28 pp [+0.68, +3.83] | +1.64 pp [−0.24, +3.49] | +0.67 pp [−1.22, +2.63] |
| Middle | +1.72 pp [+0.06, +3.34] | **+4.06 pp** [+2.32, +5.78] | +1.89 pp [−0.11, +4.01] |
| Last | **+3.19 pp** [+1.60, +4.76] | −0.26 pp [−2.22, +1.66] | +0.33 pp [−1.66, +2.35] |

- The gain does not grow steadily as the memory grows.
- Urban is best in its last third, VideoSIS in its middle third, and Video-MME stays small throughout.

## 5. Memory and output format
- **Memory after the last block:**
  - Urban: 95 items (20 active, 58 hold; the rest merged or pruned)
  - VideoSIS: 138 items (20 active, 84 hold)
  - Video-MME: 73 items (19 active, 49 hold)
- **Invalid output with the example, so the vanilla answer was kept:** Urban 43, VideoSIS 114, Video-MME 49.
- **Final invalid outputs:** Urban 3, VideoSIS 3, Video-MME 16. Vanilla had 89, 14 and 66.
- **Shown examples whose correct option text equals the current question's correct option text:** 0 on all three
  benchmarks.

## 6. Interpretation
- **Urban and VideoSIS:** the memory holds many similar questions. The median similarity of the shown example is
  0.74 and 0.71.
  - Procedures such as counting, route progress, landmark recall and direction judgments transfer.
  - Low-confidence questions gain about 4–5 pp.
  - Gains concentrate in procedural, recurring question types: Urban's Action Generation and Progress Evaluation;
    VideoSIS's Path Planning, Landmark Recall and Action Recall (see `subset_analysis.md`).
- **Video-MME:** its videos are diverse, so even the top-1 example is less similar (median similarity 0.49).
  - Apart from fixing broken output formats, there is little gain.
  - By task type, Object Reasoning (+16), Information Synopsis (+11) and OCR (+7) improve.
  - Action Recognition (−12) gets worse, as it also does on VideoSIS.
- About 55–62% of questions are confident and get no example. They cannot be harmed, and extra inference is needed
  only for the remaining 38–45%.

## 7. Limitations
- **The 0.99 threshold was chosen on the same questions**, using a different (frozen) memory. Thresholds between
  0.95 and 0.995 differed by less than 0.3 pp on average.
- **Urban and VideoSIS results were seen in earlier experiments.**
- **Some option letters inside examples did not match the shuffled option order.** For example, an example says
  "D is the only beef option" while its answer slot shows B, because only `option D`, `(D)` and `D.` are remapped.
  The same issue was present in earlier experiments.
- **The critic prompts misstate how retrieval works.** They say that items are retrieved by `when_to_use` and that
  held items are never shown. In fact the run retrieved by the source question's vector and also showed held items.
- **The only control is vanilla.** No random-example or frozen-memory condition was run, so we cannot tell whether
  the format-repair effect comes from the example's content.
- **Teacher and critic are API models and are not deterministic.** A rerun builds a different memory.
- Fixed/broken counts and memory usage statistics are associations, not causal effects.
