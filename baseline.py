from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

benchmark_items = []
with open("data/engineering_reasoning_benchmark_v0_1.jsonl", "r") as file:
    for line in file:
        benchmark_items.append(json.loads(line))

max_new_tokens = 1000
model.eval()

with open("results/baseline_results.jsonl", "w") as output_file:

    for benchmark_item in benchmark_items:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": benchmark_item["prompt"]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

        model_inputs = tokenizer(text, return_tensors = "pt").to(device)

        with torch.inference_mode():
            model_outputs = model.generate(**model_inputs, max_new_tokens = max_new_tokens, do_sample = False)

        generated_tokens = model_outputs[0][model_inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(generated_tokens, skip_special_tokens = True)

        result = {
            "id": benchmark_item["id"],
            "response": response,
            "generated_tokens": len(generated_tokens),
            "truncated": len(generated_tokens) == max_new_tokens
        }

        print(f"Benchmark item {benchmark_item['id']} processed. Generated tokens: {result['generated_tokens']}. Truncated: {result['truncated']}")

        output_file.write(json.dumps(result) + "\n")