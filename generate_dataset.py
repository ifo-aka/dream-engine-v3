import json
import urllib.request
import os
import re

ALPACA_FILE = "alpaca.json"
DOLLY_FILE = "dolly.jsonl"
OUTPUT_FILE = "dataset.txt"

def clean_content(text):
    if not text: return ""
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    # Strip any existing tags if present to prevent double tagging
    text = text.replace("<|user|>", "").replace("<|assistant|>", "").replace("<|eos|>", "").replace("<|bos|>", "")
    return text.strip()

print(f"Checking datasets...")

# 1. Download Alpaca if missing
if not os.path.exists(ALPACA_FILE):
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, ALPACA_FILE)
    except Exception as e:
        print(f"Failed to download Alpaca: {e}")

# 2. Download Dolly 15k if missing
if not os.path.exists(DOLLY_FILE):
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, DOLLY_FILE)
    except Exception as e:
        print(f"Failed to download Dolly: {e}")

entries_written = 0
total_size = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # 1. Base Greetings
    greetings = [
        ("Hi", "Hello! I am your AI assistant. How can I help you today?"),
        ("Hello", "Hello there! I'm ready to assist you. What's on your mind?"),
        ("Who are you?", "I am a Transformer-based AI trained to follow instructions and answer questions."),
    ]
    for u, a in greetings:
        f.write(f"<|bos|>\n<|user|>\n{u}\n<|assistant|>\n{a}\n<|eos|>\n")

    # 2. Process Alpaca
    if os.path.exists(ALPACA_FILE):
        print("Processing Alpaca...")
        with open(ALPACA_FILE, "r", encoding="utf-8") as af:
            alpaca_data = json.load(af)
            for entry in alpaca_data:
                instruction = clean_content(entry.get("instruction", ""))
                input_text = clean_content(entry.get("input", ""))
                output = clean_content(entry.get("output", ""))
                
                if len(instruction) < 5 or len(output) < 10: continue
                
                user_prompt = instruction
                if input_text: user_prompt += "\n" + input_text
                
                f.write(f"<|bos|>\n<|user|>\n{user_prompt}\n<|assistant|>\n{output}\n<|eos|>\n")
                entries_written += 1

    # 3. Process Dolly
    if os.path.exists(DOLLY_FILE):
        print("Processing Dolly...")
        with open(DOLLY_FILE, "r", encoding="utf-8") as df:
            for line in df:
                try:
                    entry = json.loads(line)
                    instruction = clean_content(entry.get("instruction", ""))
                    context = clean_content(entry.get("context", ""))
                    response = clean_content(entry.get("response", ""))
                    
                    if len(instruction) < 5 or len(response) < 10: continue

                    user_prompt = instruction
                    if context: user_prompt += "\nContext:\n" + context
                    
                    f.write(f"<|bos|>\n<|user|>\n{user_prompt}\n<|assistant|>\n{response}\n<|eos|>\n")
                    entries_written += 1
                except: continue

print(f"Successfully generated {OUTPUT_FILE} with {entries_written} conversation pairs.")
size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Current file size: {size_mb:.2f} MB")

# Sanity check output
print("\n--- DATA SANITY CHECK (First 200 chars) ---")
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    print(f.read(200))
