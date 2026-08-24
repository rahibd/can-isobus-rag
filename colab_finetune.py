"""
CAN/ISOBUS Fine-Tuning - Colab (free T4 GPU) version
========================================================
Fine-tunes a 7-8B open model on finetune_data.jsonl (produced locally by
dataset_prep.py) using Unsloth + QLoRA. Meant to run inside Google Colab
with a T4 GPU runtime -- NOT on your laptop, this needs real VRAM.

Setup in Colab:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload this file, then: !python colab_finetune.py
     (or paste each section into its own cell and run interactively)
  3. When prompted, upload finetune_data.jsonl from your laptop

Note: file upload / download dialogs (google.colab.files) only work when
this is actually running inside a Colab session, not on a local machine.
"""

import glob
import torch

# -----------------------------
# 1. CONFIRM GPU RUNTIME
# -----------------------------
assert torch.cuda.is_available(), (
    "No GPU detected -- go to Runtime > Change runtime type > T4 GPU, then re-run."
)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# NOTE: Unsloth itself is installed via pip BEFORE running this script:
#   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from google.colab import files


# -----------------------------
# 2. UPLOAD YOUR DATASET
# -----------------------------
print("\nSelect finetune_data.jsonl (from dataset_prep.py on your laptop)...")
uploaded = files.upload()
DATA_FILE = list(uploaded.keys())[0]
print(f"Uploaded: {DATA_FILE}")

raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"{len(raw_dataset)} QA pairs loaded")
print(raw_dataset[0])


# -----------------------------
# 3. LOAD BASE MODEL (4-BIT)
# -----------------------------
# Swap MODEL_NAME for any tag from https://huggingface.co/unsloth if you
# want a different size -- e.g. "unsloth/Llama-3.2-3B-Instruct" for a
# faster/smaller run, or a Qwen2.5-7B-Instruct variant for a bigger one.
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,           # auto-detect best dtype for the T4
    load_in_4bit=True,    # ~70% VRAM reduction, this is the "QLoRA" part
)


# -----------------------------
# 4. FORMAT DATASET WITH CHAT TEMPLATE
# -----------------------------
def format_example(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


dataset = raw_dataset.map(format_example)
print(dataset[0]["text"])


# -----------------------------
# 5. ATTACH LORA ADAPTER
# -----------------------------
# Only ~1% of parameters get trained here -- the base model's own weights
# stay frozen the whole time.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's memory-optimized version
    random_state=3407,
)


# -----------------------------
# 6. TRAIN
# -----------------------------
# Roughly 15-40 min on a free T4 for ~500-800 examples, 3 epochs -- exact
# time depends on how busy Colab's shared GPUs are.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    ),
)

trainer.train()


# -----------------------------
# 7. SAVE + DOWNLOAD THE LORA ADAPTER
# -----------------------------
model.save_pretrained("can_isobus_lora")
tokenizer.save_pretrained("can_isobus_lora")

import subprocess
subprocess.run(["zip", "-r", "can_isobus_lora.zip", "can_isobus_lora"], check=True)
files.download("can_isobus_lora.zip")


# -----------------------------
# 8. OPTIONAL: EXPORT TO GGUF FOR LM STUDIO
# -----------------------------
# Since your existing RAG project already runs on LM Studio, exporting to
# GGUF lets you load the fine-tuned model exactly like gemma4:e4b now --
# no separate serving setup needed. Comment this section out if you only
# want the raw LoRA adapter.
model.save_pretrained_gguf("can_isobus_gguf", tokenizer, quantization_method="q4_k_m")

gguf_path = glob.glob("can_isobus_gguf/*.gguf")[0]
print(f"GGUF file: {gguf_path}")
files.download(gguf_path)

print("\nDone. Next: in LM Studio, My Models -> Import Model File -> point at "
      "the downloaded .gguf, then load it in the Local Server tab just like "
      "gemma4:e4b. Ask it the same questions you tested against your RAG "
      "pipeline -- WITHOUT giving it retrieved context -- to see what it "
      "actually learned versus what RAG was supplying at question time.")
