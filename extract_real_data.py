"""
extract_real_data.py
Extracts verified real tasks and interaction trajectories from Mind2Web.
Preserves original annotation IDs, instructions, websites, domains, and human-annotated actions.
Discards raw and cleaned HTML dumps to keep the dataset lightweight and memory-efficient.
"""

import json
import os
import sys
import time
from datetime import datetime

def extract_mind2web_tasks(target_count=75, output_file="real_mind2web_tasks.json"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Hugging Face datasets: osunlp/Mind2Web (streaming mode)...")
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    extracted_tasks = []
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Streaming samples and stripping heavy HTML dumps...")
    
    for idx, item in enumerate(ds):
        # Extract metadata
        annotation_id = item["annotation_id"]
        confirmed_task = item["confirmed_task"]
        website = item.get("website", "")
        domain = item.get("domain", "")
        subdomain = item.get("subdomain", "")
        action_reprs = item.get("action_reprs", [])
        
        # Extract action sequence without HTML dumps
        raw_actions = item.get("actions", [])
        structured_actions = []
        for act in raw_actions:
            op_dict = act.get("operation", {})
            structured_actions.append({
                "action_uid": act.get("action_uid", ""),
                "op": op_dict.get("op", ""),
                "original_op": op_dict.get("original_op", ""),
                "value": op_dict.get("value", "")
            })
            
        record = {
            "dataset_name": "Mind2Web",
            "task_id": annotation_id,
            "trajectory_id": f"{annotation_id}_traj",
            "source_split": "train",
            "website": website,
            "domain": domain,
            "subdomain": subdomain,
            "confirmed_task": confirmed_task,
            "action_reprs": action_reprs,
            "actions": structured_actions
        }
        
        extracted_tasks.append(record)
        if (idx + 1) % 10 == 0 or (idx + 1) == target_count:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracted {len(extracted_tasks)}/{target_count} real tasks... (Elapsed: {time.time()-t0:.1f}s)")
            
        if len(extracted_tasks) >= target_count:
            break

    total_time = time.time() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully extracted {len(extracted_tasks)} real tasks in {total_time:.2f}s.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_tasks, f, indent=2)
        
    print(f"Saved real dataset to {output_file} ({os.path.getsize(output_file) / 1024:.1f} KB).")
    return extracted_tasks

if __name__ == "__main__":
    count = 70
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    extract_mind2web_tasks(target_count=count)
