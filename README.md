# Evaluating and Fine-Tuning Engineering Reasoning in a Small Language Model

## Overview

Large language models can produce plausible engineering explanations while making subtle errors in physical reasoning. This project investigates whether targeted supervised fine-tuning can improve engineering reasoning in a small open-source language model and how reliably those improvements can be measured.

I developed a 16-question engineering reasoning benchmark, validated an LLM-based evaluator against expert annotations, created a targeted fine-tuning dataset, and fine-tuned Qwen2.5-1.5B-Instruct using LoRA.

At the aggregate level, fine-tuning did **not** improve the automated benchmark score: the base model scored **26/64** and the fine-tuned model scored **25/64**. However, expert review of all 16 response pairs found that:

* 6 showed meaningful qualitative improvement
* 2 showed minor improvement
* 7 showed no meaningful change
* 1 showed a clear regression

Several improved responses remained incomplete or incorrect. The experiment therefore suggests that fine-tuning changed the model's engineering reasoning behavior without reliably improving overall correctness—and that coarse automated metrics did not capture all of those changes.

---

## Research Questions

1. **Can targeted supervised fine-tuning improve the physical reasoning of a small instruction-tuned language model on engineering mechanics problems?**
2. **How reliably can an LLM-based evaluator detect changes in engineering reasoning?**

The second question became increasingly important because changes in reasoning quality did not always correspond to changes in benchmark score.

---

## Experimental Design

The experiment used four components:

1. A held-out engineering reasoning benchmark.
2. An LLM evaluator validated against expert judgment.
3. A targeted supervised fine-tuning dataset.
4. A controlled comparison between the base and fine-tuned models.

The benchmark prompts, generation settings, rubric, and evaluator were held constant before and after fine-tuning.

### Base Model

**Qwen2.5-1.5B-Instruct**

The small model size made local experimentation and fine-tuning practical while still providing a capable instruction-tuned model for studying reasoning behavior.

---

## Engineering Reasoning Benchmark

The benchmark contains **16 held-out engineering mechanics questions** targeting common reasoning failures rather than simple computational errors.

Examples include:

* confusing zero velocity with zero acceleration,
* treating static friction as automatically equal to μN,
* misapplying the two-force member assumption,
* failing to check rigid-body assumptions,
* misinterpreting centripetal force or acceleration,
* and reasoning incorrectly about constraint forces or free body diagrams.

Each item contains a reference answer, reference reasoning, targeted failure mode, and scoring rubric.

### Scoring

| Criterion                      | Score |
| ------------------------------ | ----: |
| Final conclusion               |   0–1 |
| Physical reasoning             |   0–2 |
| Assumption/constraint handling |   0–1 |
| **Maximum**                    | **4** |

The evaluator also records whether the targeted failure mode was exhibited.

This separates getting the right answer from reaching it through physically valid reasoning.

---

## Automated Evaluation

Responses were evaluated using **GPT-OSS-120B through the Groq API**.

The evaluator receives the problem, reference answer, reference reasoning, targeted failure mode, rubric, and model response, then returns structured scores and a brief justification. Total score is calculated separately in Python.

### Judge Validation

Before using the evaluator for the fine-tuning comparison, I compared its ratings against expert annotations.

Held-out agreement was:

* **94.2%** across all rubric dimensions
* **94.9%** across the three scored dimensions

Validation also revealed two recurring limitations: the judge occasionally overestimated physical-reasoning quality and sometimes interpreted failure modes too literally. The evaluator configuration was then frozen before evaluating the fine-tuned model.

---

## Fine-Tuning Dataset

I created **64 engineering reasoning examples** covering the same broad reasoning skills as the benchmark without duplicating benchmark questions.

The dataset spans 16 skills, including:

* rotating dynamics,
* centripetal force,
* projectile-motion assumptions,
* static friction,
* constraint forces,
* two-force members,
* rigid-body assumptions,
* rolling contact,
* free-body diagrams,
* circular motion,
* and equilibrium reasoning.

The data were split into:

* **48 training examples**
* **16 validation examples**

Training responses were structured to encourage the model to:

1. State the conclusion.
2. Identify the relevant physical principle, assumption, or constraint.
3. Apply it to the specific problem.

---

## LoRA Fine-Tuning

The base model was fine-tuned using supervised fine-tuning with LoRA.

| Parameter               | Value                 |
| ----------------------- | --------------------- |
| LoRA rank               | 8                     |
| LoRA alpha              | 16                    |
| Target modules          | `q_proj`, `v_proj`    |
| LoRA dropout            | 0.05                  |
| Learning rate           | 1e-4                  |
| Epochs                  | 3                     |
| Batch size              | 4                     |
| Maximum sequence length | 256                   |
| Loss                    | Assistant tokens only |

Trainable parameters:

**1,089,536 / 1,544,803,840 = 0.0705%**

Validation loss decreased:

**2.140 → 2.064 → 2.048**

Validation token accuracy increased:

**0.504 → 0.525 → 0.532**

These metrics showed that the model learned the fine-tuning distribution, but not whether engineering reasoning improved on the held-out benchmark.

---

## Results

The base and fine-tuned models were evaluated using identical prompts, deterministic generation settings, and the frozen evaluator.

| Metric                   |      Base | Fine-Tuned |
| ------------------------ | --------: | ---------: |
| Total score              | **26/64** |  **25/64** |
| Average score            |   1.625/4 |    1.563/4 |
| Correct conclusions      |      7/16 |       8/16 |
| Final-answer accuracy    |    43.75% |     50.00% |
| Avg. physical reasoning  |   0.813/2 |    0.750/2 |
| Avg. assumption handling |   0.375/1 |    0.313/1 |
| Failure-mode rate        |    43.75% |     37.50% |

At the aggregate level, fine-tuning did **not** improve benchmark performance.

Because the benchmark contains only 16 questions, each problem represents 6.25 percentage points of final-answer accuracy. These differences should therefore be interpreted cautiously.

---

## Expert Error Analysis

The aggregate result raised a more important question:

**Did fine-tuning actually make the reasoning worse, or was the rubric failing to capture some of the behavioral changes?**

I manually compared all 16 response pairs.

| Classification                     | Responses |
| ---------------------------------- | --------: |
| Meaningful qualitative improvement |         6 |
| Minor qualitative improvement      |         2 |
| No meaningful change               |         7 |
| Clear regression                   |         1 |

Thus, **8 of 16 responses showed some qualitative improvement**, although several remained incomplete or ultimately incorrect. This result is not equivalent to saying that fine-tuning improved performance on 50% of the benchmark. The expert categories capture reasoning changes that are finer-grained than the rubric.

### Selected Examples

#### AVJ_02 — Better reasoning, same score

The fine-tuned model correctly avoided assuming a magnitude for static friction, eliminating an important baseline misconception. However, it failed to complete the analysis using Newton's second law.

**Automated score: 0 → 0**

This finding illustrates how reasoning can improve without crossing a rubric threshold.

#### PMS_03 — Substantial improvement, incomplete correction

The fine-tuned model reached the correct conclusion and provided much better physical reasoning, but still failed to recognize an important applicability condition for the two-force member assumption.

**Automated score: 0 → 2**

#### AVJ_03 — Better conceptual reasoning, wrong conclusion

The fine-tuned model showed a stronger understanding of static versus kinetic friction but ultimately applied that reasoning incorrectly and reached the wrong conclusion.

**Automated score: 2 → 0**

This result shows why reasoning quality and task correctness cannot always be treated as equivalent.

#### KCM_04 — Valid alternative reasoning

The reference solution used momentum, while the fine-tuned model used Newton's second law and reached the correct conclusion through a physically valid alternative path.

The automated evaluator gave credit for the conclusion but not the physical reasoning.

This exposed a potential **reference-answer bias** in the evaluation process.

#### SDR_01 — Genuine regression

The fine-tuned model introduced an unsupported constant radial speed assumption, later conflated constant radial velocity with zero radial velocity, and failed to apply the relevant radial force constraint. This result represents a genuine reasoning regression.

---

## What Changed After Fine-Tuning?

A recurring pattern was that the fine-tuned model sometimes improved at **local physical reasoning**. For example, discussing friction more appropriately or selecting a more relevant principle while still struggling with higher-level questions such as:

* whether an assumption was valid,
* whether a constraint was correctly applied,
* or whether the reasoning actually supported the conclusion.

This finding suggests a possible distinction between learning engineering reasoning patterns and learning **when those patterns are applicable**. The experiment is too small to establish that as a general result, but it provides a clear direction for future work.

---

## Key Findings

1. **LoRA fine-tuning changed model behavior but did not improve aggregate benchmark score.**
2. **Final-answer accuracy increased slightly, while automated reasoning scores decreased slightly.**
3. **Expert review found some qualitative improvement in 8 of 16 responses, though several remained incorrect or incomplete.**
4. **Seven responses were essentially unchanged and one clearly regressed.**
5. **The automated rubric missed some partial improvements and under-credited at least one valid alternative solution path.**
6. **Fine-tuning appeared more effective at changing local reasoning patterns than at enforcing when assumptions and models were applicable.**
7. **Evaluating open-ended engineering reasoning proved to be a substantial technical challenge in its own right.**

---

## Limitations

This is a small exploratory experiment. Key limitations include:

* 16 benchmark questions,
* 64 fine-tuning examples,
* one base model,
* one LoRA configuration,
* one primary automated evaluator,
* one domain expert conducting the qualitative review,
* and a benchmark focused mainly on undergraduate mechanics.

The results should therefore be interpreted as observations about model behavior rather than statistically robust estimates of general engineering reasoning performance.

---

## Future Work

Possible next steps include:

* expanding the held-out benchmark,
* increasing training-data diversity,
* explicitly training applicability-condition reasoning,
* improving evaluator robustness to alternative valid derivations,
* testing different model sizes and LoRA configurations,
* and exploring preference optimization or reinforcement-learning approaches.

These are left for future iterations rather than modifying the original experiment after observing its results.

---

## Repository Structure

```text
data/
├── engineering_reasoning_benchmark_v0_1.jsonl
├── engineering_reasoning_finetuning_v0_1.jsonl
├── train_v0_1.jsonl
└── validation_v0_1.jsonl

results/
├── baseline_results.jsonl
├── baseline_evaluations.jsonl
├── expert_annotations.jsonl
├── finetuned_results_v0_1.jsonl
└── finetuned_evaluations_v0_1.jsonl

baseline.py
prepare_finetuning_data.py
train.py
finetuned_baseline.py
evaluate.py
analyze_results.py
```

---

## Summary

This project began as an experiment in improving engineering reasoning through small-scale supervised fine-tuning. The aggregate result was simple:

**26/64 before fine-tuning and 25/64 afterward.**

Closer analysis showed a more interesting pattern. Some responses became more physically informed, others remained unchanged, and one clearly regressed. Several of those reasoning changes were not captured by the automated rubric. The main lesson is that **changes in technical reasoning can be harder to measure than changes in final-answer accuracy**. For engineering applications of language models, improving the model and designing an evaluation capable of detecting that improvement are tightly coupled problems.
