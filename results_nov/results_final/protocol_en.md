# Final Full-Test Protocol (2026-11-01, written before any result was seen)

- Config: `configs/final_c.yaml`. The exact config of the reported run is `config_used.json`.
- Run: `scripts/run_icl_confidence.py --config configs/final_c.yaml`.
- This is an English translation of `protocol.md`.

## 1. Comparison
- **A. Vanilla:** the existing answers without examples. Urban and Video-MME come from the earlier full vanilla run;
  VideoSIS comes from the vanilla answers of the earlier VideoSIS run.
- **C. Confidence gate + evolving memory:** the rules below.
- A and C answer the same questions in the same order, and are compared question by question.
- Benchmarks:
  - Urban Video Bench: 5,355 questions, 108 blocks
  - VideoSIS: 4,852 questions, 98 blocks
  - Video-MME: 2,700 questions, 54 blocks

  The three run side by side, each with its own memory.

## 2. Answering one question
1. Take the vanilla answer and its confidence p1.
   - p1 is the probability of the answer-letter token (`>X`), renormalised over the question's own option letters;
     we take the maximum.
   - It comes from a forward pass on the unchanged vanilla input (`answer_probs.sqlite`).
2. If p1 ≥ 0.99, the vanilla answer is final. The input is byte-identical to vanilla, so the answer is too.
3. If p1 < 0.99, or p1 cannot be read (no option letter right after `<ANSWER>` in the vanilla output), show one
   example:
   - Candidates are the benchmark memory's active + hold items.
   - Exclude items from the same original video.
   - Exclude items whose correct option text equals one of the current options. This uses no label.
   - Take the single item whose source question has the highest WeMM query-vector cosine similarity.
     **There is no similarity floor.**
   - The example's options are shuffled at random, and the example is placed after the video and before the new
     question.
4. If the second answer is invalid, keep the vanilla answer. An answer is invalid when there is not exactly one
   `<ANSWER>` block, or its content is not an option letter.
   - Outputs are never edited or regenerated.
   - If the memory is empty, or no item qualifies, the vanilla answer is used.
5. The 0.99 threshold was chosen on 2026-11-01 from a threshold analysis with a frozen sol memory. That analysis
   file is not included in this repository.
   - The threshold was chosen on the same questions, and results must state this.
   - Thresholds between 0.95 and 0.995 differed by less than 0.3 pp on average.

## 3. Memory evolution (every block of 50 questions, as in the earlier evolve tracks)
- The memory starts empty. Block 1 is answered without examples.
- Labels are used **only after** all answers of the block are fixed. A question's label and the teacher's solution
  enter the memory only after that question has been answered.
- Five stages. Teacher and critic are both gpt-6.1-sol.
  1. The critic selects failure cases, and each case is marked low-confidence or not.
  2. The teacher watches the frames and solves the cases without the answer.
  3. The critic compares with the student's output and writes example candidates.
  4. The student answers up to 12 past low-confidence questions from other videos with each candidate attached. The
     critic revises the candidates from those results, at most twice.
  5. The critic adds, merges, prunes and refines memory items, using each item's fixed / broken record. At most 20
     items are active.
- **Example length limit:** worked_output ≤ 1,300 characters (about 1,800 with the question and options). The limit
  is stated in the prompts, and longer outputs are rejected.

## 4. Cost and stopping
- sol prices per 1M tokens: input $2.00, cached input $0.10, output $10.00.
- At a cumulative $79.5 no new call is made. The cap is $80; parallel calls may exceed it by about $0.3.
- A stopped benchmark is reported up to its last finished block.
- A restart resumes after the last finished block. Identical calls are served from the cache at no new cost.

## 5. Reporting
Results go to `results_nov/results_final/<benchmark>/`, updated every 5 blocks and when the benchmark ends.
- Files:
  - `model_output.jsonl` (the first line is an accuracy summary), `memory.jsonl`, `block_curve.jsonl`,
    `summary.json` and `summary.md`;
  - `overall.jsonl`, with one summary per benchmark.
- C vs A: accuracy difference, video-cluster bootstrap 95% CI, fixed / broken counts, and a sign test.
- Report separately the questions given an example, the low-confidence questions with no qualifying item, and the
  confident questions. Also report results by confidence band and the block curve.
- Report the number of shown examples whose correct option text equals the current question's correct option text.
- Memory usage statistics are associations, not causal effects.
