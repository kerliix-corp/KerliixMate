# scripts/prepare_data.py
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import random

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(items, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def combine_prompt_system(item):
    """
    Combine system + prompt into single prompt string used for SFT.
    We use a template so the tokenizer learns the structure.
    """
    sys = item.get("system", "").strip()
    prompt = item.get("prompt", "").strip()
    if sys:
        return f"<|system|>\n{sys}\n<|end_system|>\n<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

def main(args):
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = list(load_jsonl(src))
    # attach a combined sequence
    for it in all_items:
        it["sft_input"] = combine_prompt_system(it)
        # target is the response plus an end marker
        it["sft_target"] = it.get("response", "").strip() + "\n<|endofresponse|>"

    train, val = train_test_split(all_items, test_size=args.val_ratio, random_state=42)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    print(f"Wrote {len(train)} train and {len(val)} val examples to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source jsonl file")
    parser.add_argument("--out_dir", default="data/processed", help="output folder")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
