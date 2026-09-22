import json

examples = []

with open("data/engineering_reasoning_finetuning_v0_1.jsonl", "r") as file:
    for line in file:
        examples.append(json.loads(line))

examples_by_skill = {}

for example in examples:
    skill = example["skill"]

    if skill not in examples_by_skill:
        examples_by_skill[skill] = []

    examples_by_skill[skill].append(example)

train_examples = []
validation_examples = []

for skill, skill_examples in examples_by_skill.items():
    train_examples.extend(skill_examples[:3])
    validation_examples.append(skill_examples[3])

print("Training examples:", len(train_examples))
print("Validation examples:", len(validation_examples))

with open("data/train_v0_1.jsonl", "w") as file:
    for example in train_examples:
        file.write(json.dumps(example) + "\n")

with open("data/validation_v0_1.jsonl", "w") as file:
    for example in validation_examples:
        file.write(json.dumps(example) + "\n")