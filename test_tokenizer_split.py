import os
import sys
import json
import random
import time
import math
import re
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# 1. Generate workflows
from test_generator import generate_synthetic_workflows, DOMAINS

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

workflows = generate_synthetic_workflows(320)
print(f"Total workflows: {len(workflows)}")

def workflow_to_text(wf):
    lines = [
        f"Goal: {wf['goal']}.",
        "",
        f"Description: {wf['description']}",
        "",
        "Nodes:"
    ]
    for n in wf['nodes']:
        lines.append(f"{n['description']}.")
    
    comp = [n['description'] for n in wf['nodes'] if n['status'] == 'COMPLETED']
    inc = [n['description'] for n in wf['nodes'] if n['status'] == 'INCOMPLETE']
    fail = [n['description'] for n in wf['nodes'] if n['status'] == 'FAILED']
    
    lines.extend(["", "Completed:"])
    if comp:
        for c in comp:
            lines.append(f"{c}.")
    else:
        lines.append("None.")
        
    lines.extend(["", "Incomplete:"])
    if inc:
        for itm in inc:
            lines.append(f"{itm}.")
    else:
        lines.append("None.")
        
    lines.extend(["", "Failed:"])
    if fail:
        for f in fail:
            lines.append(f"{f}.")
    else:
        lines.append("None.")
        
    lines.extend(["", "Dependencies:"])
    id_to_desc = {n['node_id']: n['description'] for n in wf['nodes']}
    if wf.get('dependencies'):
        for src, dst in wf['dependencies']:
            if src in id_to_desc and dst in id_to_desc:
                lines.append(f"{id_to_desc[src]} -> {id_to_desc[dst]}")
    else:
        lines.append("None.")
        
    return "\n".join(lines)

for w in workflows:
    w['text_repr'] = workflow_to_text(w)

print("Sample text representation (first 200 chars):")
print(workflows[0]['text_repr'][:200])

# 2. Leakage-free splitting
all_wf_ids = sorted([w['workflow_id'] for w in workflows])
random.shuffle(all_wf_ids)

n_total = len(all_wf_ids)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_ids = set(all_wf_ids[:n_train])
val_ids = set(all_wf_ids[n_train:n_train + n_val])
test_ids = set(all_wf_ids[n_train + n_val:])

assert train_ids.isdisjoint(val_ids), "Train and Val overlap!"
assert train_ids.isdisjoint(test_ids), "Train and Test overlap!"
assert val_ids.isdisjoint(test_ids), "Val and Test overlap!"

wf_dict = {w['workflow_id']: w for w in workflows}
train_wfs = [wf_dict[wid] for wid in sorted(train_ids)]
val_wfs = [wf_dict[wid] for wid in sorted(val_ids)]
test_wfs = [wf_dict[wid] for wid in sorted(test_ids)]

print(f"Train workflows: {len(train_wfs)}, Val: {len(val_wfs)}, Test: {len(test_wfs)}")

# 3. Custom Tokenizer
class CustomTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    
    PAD_ID = 0
    UNK_ID = 1
    START_ID = 2
    END_ID = 3
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.START_TOKEN: self.START_ID,
            self.END_TOKEN: self.END_ID
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.is_built = False
        
    @staticmethod
    def clean_and_tokenize(text):
        text = text.lower()
        # Separate words and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            tokens = self.clean_and_tokenize(text)
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
        
        for tok, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if count >= self.min_freq and tok not in self.word2idx:
                new_id = len(self.word2idx)
                self.word2idx[tok] = new_id
                self.idx2word[new_id] = tok
                
        self.is_built = True
        return len(self.word2idx)
        
    def encode(self, text, max_len=None, add_special_tokens=True):
        tokens = self.clean_and_tokenize(text)
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.START_ID)
        for tok in tokens:
            token_ids.append(self.word2idx.get(tok, self.UNK_ID))
        if add_special_tokens:
            token_ids.append(self.END_ID)
            
        if max_len is not None:
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                if add_special_tokens:
                    token_ids[-1] = self.END_ID
            mask = [1] * len(token_ids)
            while len(token_ids) < max_len:
                token_ids.append(self.PAD_ID)
                mask.append(0)
            return token_ids, mask
        else:
            mask = [1] * len(token_ids)
            return token_ids, mask

    def decode(self, token_ids):
        tokens = []
        for tid in token_ids:
            if tid in self.idx2word:
                tok = self.idx2word[tid]
                if tok not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                    tokens.append(tok)
        return " ".join(tokens)

# Build vocab STRICTLY on training split
train_texts = []
for w in train_wfs:
    train_texts.append(w['text_repr'])
    train_texts.extend(w['queries'])

tokenizer = CustomTokenizer(min_freq=1)
vocab_size = tokenizer.build_vocab(train_texts)
print(f"Vocabulary size (trained ONLY on training set): {vocab_size}")

test_encode_ids, test_mask = tokenizer.encode("Continue my Python REST API work", max_len=16)
print(f"Test encode IDs: {test_encode_ids}")
print(f"Test decode: {tokenizer.decode(test_encode_ids)}")
