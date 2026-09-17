from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [ 
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"}
]

text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

model_inputs = tokenizer(text, return_tensors = "pt")

model_outputs = model.generate(**model_inputs, max_new_tokens = 100)

generated_tokens = model_outputs[0][model_inputs["input_ids"].shape[1]:]

response = tokenizer.decode(generated_tokens, skip_special_tokens = True)
print(response)