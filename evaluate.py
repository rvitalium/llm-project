import json
from groq import Groq

benchmark_items = []
with open("data/engineering_reasoning_benchmark_v0_1.jsonl", "r") as file:
    for line in file:
        benchmark_items.append(json.loads(line))

baseline_results = []
with open("results/baseline_results.jsonl", "r") as file:
    for line in file:
        baseline_results.append(json.loads(line))

assert benchmark_items[0]["id"] == baseline_results[0]["id"], "IDs do not match between benchmark items and baseline results."

benchmark_item = benchmark_items[0]
baseline_result = baseline_results[0]

judge_prompt = f"""
You are evaluating the response of an AI model on an engineering reasoning benchmark.

Apply the provided scoring rubric exactly. Do not give credit for reasoning that is physically incorrect, even if the final answer happens to be correct.

ORIGINAL PROMPT:
{benchmark_item["prompt"]}

CORRECT ANSWER:
{benchmark_item["correct_answer"]}

REFERENCE REASONING:
{benchmark_item["why"]}

TARGETED FAILURE MODE:
{benchmark_item["failure_mode"]}

SCORING RUBRIC:
{json.dumps(benchmark_item["scoring_rubric"], indent=2)}

MODEL RESPONSE:
{baseline_result["response"]}

Evaluate the model response dimension by dimension.

Return:
1. final_conclusion score
2. physical_reasoning score
3. assumption_constraint_handling score
4. whether the targeted failure mode was triggered: true or false
5. a brief justification for each judgment

Do not calculate a total score.
"""

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

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": judge_prompt}
    ],
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

print(total_score)