import json

baseline_results = []
with open("results/baseline_evaluations.jsonl", "r") as file:
    for line in file:
        baseline_results.append(json.loads(line))

finetuned_results = []
with open("results/finetuned_evaluations_v0_1.jsonl", "r") as file:
    for line in file:
        finetuned_results.append(json.loads(line))

total_score = 0
correct_conclusions = 0
total_physical_reasoning = 0
failure_mode_count = 0
total_assumption_handling = 0

for result in baseline_results:
    total_score += result["total_score"]
    correct_conclusions += result["final_conclusion"]
    total_physical_reasoning += result["physical_reasoning"]
    total_assumption_handling += result["assumption_constraint_handling"]
    failure_mode_count += result["failure_mode_trigger"]

average_score = total_score / len(baseline_results) if baseline_results else 0
average_physical_reasoning = total_physical_reasoning / len(baseline_results) if baseline_results else 0
conclusion_accuracy = correct_conclusions / len(baseline_results) if baseline_results else 0
failure_mode_rate = (failure_mode_count / len(baseline_results) if baseline_results else 0)
average_assumption_handling = (total_assumption_handling / len(baseline_results)if baseline_results else 0)

print('BASELINE RESULTS')

print("Total score:", total_score)
print("Average score:", average_score)

print("Correct final conclusions:", correct_conclusions)
print("Final conclusion accuracy:", conclusion_accuracy)

print("Total physical reasoning:", total_physical_reasoning)
print("Average physical reasoning:", average_physical_reasoning)

print("Failure modes triggered:", failure_mode_count)
print("Failure mode trigger rate:", failure_mode_rate)

print("Total assumption/constraint score:", total_assumption_handling)
print("Average assumption/constraint score:", average_assumption_handling)

total_score = 0
correct_conclusions = 0
total_physical_reasoning = 0
failure_mode_count = 0
total_assumption_handling = 0

for result in finetuned_results:
    total_score += result["total_score"]
    correct_conclusions += result["final_conclusion"]
    total_physical_reasoning += result["physical_reasoning"]
    total_assumption_handling += result["assumption_constraint_handling"]
    failure_mode_count += result["failure_mode_trigger"]


average_score = total_score / len(baseline_results) if baseline_results else 0
average_physical_reasoning = total_physical_reasoning / len(baseline_results) if baseline_results else 0
conclusion_accuracy = correct_conclusions / len(baseline_results) if baseline_results else 0
failure_mode_rate = (failure_mode_count / len(baseline_results) if baseline_results else 0)
average_assumption_handling = (total_assumption_handling / len(baseline_results)if baseline_results else 0)

print('FINE TUNED RESULTS')

print("Total score:", total_score)
print("Average score:", average_score)

print("Correct final conclusions:", correct_conclusions)
print("Final conclusion accuracy:", conclusion_accuracy)

print("Total physical reasoning:", total_physical_reasoning)
print("Average physical reasoning:", average_physical_reasoning)

print("Failure modes triggered:", failure_mode_count)
print("Failure mode trigger rate:", failure_mode_rate)

print("Total assumption/constraint score:", total_assumption_handling)
print("Average assumption/constraint score:", average_assumption_handling)

for baseline, finetuned in zip(baseline_results, finetuned_results):
    score_change = finetuned["total_score"] - baseline["total_score"]

    print()
    print(baseline["id"])
    print("Baseline:", baseline["total_score"])
    print("Fine-tuned:", finetuned["total_score"])
    print("Change:", score_change)

for baseline, finetuned in zip(baseline_results, finetuned_results):
    print()
    print(baseline["id"])

    print(
        "Final conclusion:",
        baseline["final_conclusion"],
        "->",
        finetuned["final_conclusion"]
    )

    print(
        "Physical reasoning:",
        baseline["physical_reasoning"],
        "->",
        finetuned["physical_reasoning"]
    )

    print(
        "Assumption/constraint:",
        baseline["assumption_constraint_handling"],
        "->",
        finetuned["assumption_constraint_handling"]
    )

    print(
        "Failure mode:",
        baseline["failure_mode_trigger"],
        "->",
        finetuned["failure_mode_trigger"]
    )