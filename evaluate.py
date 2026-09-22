import json
from groq import Groq
import time

benchmark_items = []
with open("data/engineering_reasoning_benchmark_v0_1.jsonl", "r") as file:
    for line in file:
        benchmark_items.append(json.loads(line))

finetuned_results = []
with open("results/finetuned_results_v0_1.jsonl", "r") as file:
    for line in file:
        finetuned_results.append(json.loads(line))

score_schema = {
    "type": "object",
    "properties": {
        "final_conclusion": {
            "type": "integer",
            "enum": [0, 1]
        },
        "physical_reasoning": {
            "type": "integer",
            "enum": [0, 1, 2]
        },
        "assumption_constraint_handling": {
            "type": "integer",
            "enum": [0, 1]
        },
        "failure_mode_trigger": {
            "type": "boolean"
        },
        "justification": {
            "type": "string"
        }
    },
    "required": [
        "final_conclusion",
        "physical_reasoning",
        "assumption_constraint_handling",
        "failure_mode_trigger",
        "justification"
    ],
    "additionalProperties": False
}

client = Groq()
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PROMPT_VERSION = "v1.0"
JUDGE_REASONING_EFFORT = "low"

evaluation_results = []

for i in range(len(benchmark_items)):
    assert benchmark_items[i]["id"] == finetuned_results[i]["id"], "IDs do not match between benchmark items and finetuned results."

    benchmark_item = benchmark_items[i]
    finetuned_result = finetuned_results[i]

    judge_prompt = f"""
        Score an AI response to an engineering reasoning problem using the provided reference information and rubric.

        Apply the rubric exactly. Judge the response's physical reasoning, not merely whether its final answer matches the reference. Do not give credit for physically incorrect reasoning.

        For assumption_constraint_handling, award the point only if the relevant assumption or constraint is both correctly identified and correctly applied in the response's reasoning. Merely mentioning or showing awareness of it is insufficient.

        PROMPT:
        {benchmark_item["prompt"]}

        REFERENCE ANSWER:
        {benchmark_item["correct_answer"]}

        REFERENCE REASONING:
        {benchmark_item["why"]}

        TARGET FAILURE MODE:
        {benchmark_item["failure_mode"]}

        RUBRIC:
        {json.dumps(benchmark_item["scoring_rubric"])}

        MODEL RESPONSE:
        {finetuned_result["response"]}

        Determine each rubric score and whether the targeted failure mode is exhibited. Briefly justify the evaluation.
        """

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        reasoning_effort=JUDGE_REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "engineering_score",
                "strict": True,
                "schema": score_schema
            }
        }
    )

    judge_response = response.choices[0].message.content

    judge_score = json.loads(judge_response)

    total_score = (
        judge_score["final_conclusion"]
        + judge_score["physical_reasoning"]
        + judge_score["assumption_constraint_handling"]
    )

    evaluation_result = {
        "id": benchmark_item["id"],
        "response": finetuned_result["response"],
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "final_conclusion": judge_score["final_conclusion"],
        "physical_reasoning": judge_score["physical_reasoning"],
        "assumption_constraint_handling": judge_score["assumption_constraint_handling"],
        "failure_mode_trigger": judge_score["failure_mode_trigger"],
        "total_score": total_score,
        "justification": judge_score["justification"]
    }

    evaluation_results.append(evaluation_result)
    time.sleep(12)


print("Number of evaluations:", len(evaluation_results))
print("Prompt tokens:", response.usage.prompt_tokens)
print("Completion tokens:", response.usage.completion_tokens)
print("Reasoning tokens:", response.usage.completion_tokens_details.reasoning_tokens)
print("Total tokens:", response.usage.total_tokens)

for result in evaluation_results:
    print()
    print(result["id"])
    print("Final conclusion:", result["final_conclusion"])
    print("Physical reasoning:", result["physical_reasoning"])
    print("Assumption/constraint:", result["assumption_constraint_handling"])
    print("Failure mode:", result["failure_mode_trigger"])
    print("Total:", result["total_score"])
    print("Justification:", result["justification"])

with open("results/finetuned_evaluations_v0_1.jsonl", "w") as file:
    for result in evaluation_results:
        file.write(json.dumps(result) + "\n")