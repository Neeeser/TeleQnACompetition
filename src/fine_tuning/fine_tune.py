

import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch

# Load your dataset from the JSON file
with open("prepared_train_data.json", "r") as file:
    data = json.load(file)

# Convert to Hugging Face dataset format
dataset = Dataset.from_list(data)

# Load the Phi-2 model and tokenizer
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Add adapter layer for fine-tuning
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'],
)
model = get_peft_model(model, peft_config)

# Set up training arguments
training_arguments = TrainingArguments(
    output_dir="./phi-2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    disable_tqdm=False,
    report_to="none",
)

# Fine-tune the model
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=2048,
    dataset_text_field="input",
    packing=False,
)

trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained("./phi-2-finetuned")
tokenizer.save_pretrained("./phi-2-finetuned")

# Evaluate the model
prompt = "You are an expert in telecommunications and 3GPP standards. Answer the following multiple-choice question based on your knowledge and expertise. Please provide only the answer choice number (1, 2, 3, 4, or 5) that best answers the question. Avoid any additional explanations or text beyond the answer choice number.\n\nWhat is the purpose of the Nmfaf_3daDataManagement_Deconfigure service operation? [3GPP Release 18]\n\n(1) To configure the MFAF to map data or analytics received by the MFAF to out-bound notification endpoints\n(2) To configure the MFAF to stop mapping data or analytics received by the MFAF to out-bound notification endpoints\n(3) To supply data or analytics from the MFAF to notification endpoints\n(4) To fetch data or analytics from the MFAF based on fetch instructions\n\nAnswer:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
generated = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
