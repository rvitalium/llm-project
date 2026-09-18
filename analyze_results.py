import json

evaluation_results = []

with open("results/baseline_evaluations.jsonl", "r") as file:
    for line in file:
        evaluation_results.append(json.loads(line))

total_score = 0
correct_conclusions = 0
total_physical_reasoning = 0
failure_mode_count = 0
total_assumption_handling = 0

for result in evaluation_results:
    total_score += result["total_score"]
    correct_conclusions += result["final_conclusion"]
    total_physical_reasoning += result["physical_reasoning"]
    total_assumption_handling += result["assumption_constraint_handling"]
    failure_mode_count += result["failure_mode_trigger"]

average_score = total_score / len(evaluation_results) if evaluation_results else 0
average_physical_reasoning = total_physical_reasoning / len(evaluation_results) if evaluation_results else 0
conclusion_accuracy = correct_conclusions / len(evaluation_results) if evaluation_results else 0
failure_mode_rate = (failure_mode_count / len(evaluation_results) if evaluation_results else 0)
average_assumption_handling = (total_assumption_handling / len(evaluation_results)if evaluation_results else 0)

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