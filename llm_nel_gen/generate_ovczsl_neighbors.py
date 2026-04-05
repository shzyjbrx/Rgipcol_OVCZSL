import os
import re
import sys
import json
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="OVCZSL Step 1: Generate Neighborhood Vocabulary")
    p.add_argument("--data_root",      type=str, default="./data/mit-states")
    p.add_argument("--save_dir",       type=str, default="./LLM/neighbors")
    p.add_argument("--model_id",       type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_retries",    type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=512)
    return p.parse_args()

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_vocab(data_root: str):
    split_dir = os.path.join(data_root, "compositional-split-natural")
    attrs_set, objs_set, pairs_set = set(), set(), set()
    for fname in ["train_pairs.txt", "val_pairs.txt", "test_pairs.txt"]:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path): continue
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    attrs_set.add(parts[0])
                    objs_set.add(parts[1])
                    pairs_set.add(f"{parts[0]} {parts[1]}")
    return sorted(attrs_set), sorted(objs_set), sorted(pairs_set)

def clean_name(name: str) -> str:
    return name.replace("_", " ").strip()

# ─────────────────────────────────────────────
# Prompt 设计 (核心逻辑)
# ─────────────────────────────────────────────
def build_prompt(node_type: str, concept: str) -> list:
    concept = clean_name(concept)
    
    if node_type == "attr":
        system = "You are a visual linguistic expert. Provide visually similar or replaceable neighbors for an ATTRIBUTE. Output STRICTLY JSON."
        user = (
            f"Target Attribute: '{concept}'\n"
            "Requirement:\n"
            "1. Generate 3 visually replaceable synonyms (synonyms).\n"
            "2. Generate 2 fine-grained states or derivative visual words (fine_grained).\n"
            "Format: {\"synonyms\": [\"word1\", \"word2\", \"word3\"], \"fine_grained\": [\"word4\", \"word5\"]}"
        )
    elif node_type == "obj":
        system = "You are a computer vision commonsense engine. Find visual concept neighbors for an OBJECT. Output STRICTLY JSON."
        user = (
            f"Target Object: '{concept}'\n"
            "Requirement:\n"
            "1. Generate 2 hypernyms (hypernyms).\n"
            "2. Generate 4 visual siblings that look similar in shape/texture (visual_siblings).\n"
            "Format: {\"hypernyms\": [\"w1\", \"w2\"], \"visual_siblings\": [\"w3\", \"w4\", \"w5\", \"w6\"]}"
        )
    else: # comp
        system = "You are a multimodal image description expert. Infer reasonable related visual phrases for a COMPOSITION. Output STRICTLY JSON."
        user = (
            f"Target Composition: '{concept}'\n"
            "Requirement:\n"
            "1. 2 holistic synonyms (holistic).\n"
            "2. 2 attribute-substituted neighbors (attr_sub) with same object.\n"
            "3. 2 object-substituted neighbors (obj_sub) with same attribute.\n"
            "Format: {\"holistic\": [\"p1\", \"p2\"], \"attr_sub\": [\"p3\", \"p4\"], \"obj_sub\": [\"p5\", \"p6\"]}"
        )

    return [
        {"role": "system", "content": system + " Output English only. No preamble."},
        {"role": "user",   "content": user},
    ]

# ─────────────────────────────────────────────
# 生成与解析
# ─────────────────────────────────────────────
def generate_and_parse(tokenizer, model, node_type, concept, args) -> Dict:
    messages = build_prompt(node_type, concept)
    
    for _ in range(args.max_retries):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, do_sample=True)
        
        response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        
        # 简单清洗并解析 JSON
        try:
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            data = json.loads(json_str)
            return data
        except:
            continue
    return {"error": "Failed after retries"}

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    attrs, objs, comps = load_vocab(args.data_root)
    
    print(f"[Model] Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # 定义三组任务
    tasks = [
        (attrs, "attr", "attr_neighbors.json"),
        (objs,  "obj",  "obj_neighbors.json"),
        (comps, "comp", "comp_neighbors.json")
    ]

    for node_list, n_type, filename in tasks:
        save_path = os.path.join(args.save_dir, filename)
        
        # 断点续跑逻辑
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                results = json.load(f)
        else:
            results = {}

        print(f"\n[Processing] {n_type} nodes to {filename}...")
        for item in tqdm(node_list):
            if item in results and "error" not in results[item]:
                continue
            
            results[item] = generate_and_parse(tokenizer, model, n_type, item, args)
            
            # 每10条保存一次，防止崩溃
            if len(results) % 10 == 0:
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)
        
        # 最终保存
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[Done] All 3 neighborhood files saved in {args.save_dir}")

if __name__ == "__main__":
    main()