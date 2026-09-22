from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train_v0_1.jsonl",
        "validation": "data/validation_v0_1.jsonl"
    }
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_example(example):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return example

formatted_dataset = dataset.map(format_example)

def add_token_length(example):
    tokens = tokenizer(
        example["text"],
        add_special_tokens=False
    )

    example["token_length"] = len(tokens["input_ids"])
    return example

tokenized_info = formatted_dataset.map(add_token_length)

train_lengths = tokenized_info["train"]["token_length"]
validation_lengths = tokenized_info["validation"]["token_length"]

MAX_SEQ_LENGTH = 256

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="outputs/qwen_engineering_lora_v0_1",
    max_length=MAX_SEQ_LENGTH,
    assistant_only_loss=True,

    learning_rate=1e-4,
    num_train_epochs=3,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    eval_strategy="epoch",
    logging_steps=5,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

trainer.model.save_pretrained(
    "outputs/qwen_engineering_lora_v0_1/final_adapter"
)

tokenizer.save_pretrained(
    "outputs/qwen_engineering_lora_v0_1/final_adapter"
)