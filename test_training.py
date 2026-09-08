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

# Import from test_generator
from test_generator import generate_synthetic_workflows, DOMAINS
from test_tokenizer_split import workflow_to_text, CustomTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

workflows = generate_synthetic_workflows(320)
for w in workflows:
    w['text_repr'] = workflow_to_text(w)

all_wf_ids = sorted([w['workflow_id'] for w in workflows])
random.shuffle(all_wf_ids)

n_total = len(all_wf_ids)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_ids = set(all_wf_ids[:n_train])
val_ids = set(all_wf_ids[n_train:n_train + n_val])
test_ids = set(all_wf_ids[n_train + n_val:])

wf_dict = {w['workflow_id']: w for w in workflows}
train_wfs = [wf_dict[wid] for wid in sorted(train_ids)]
val_wfs = [wf_dict[wid] for wid in sorted(val_ids)]
test_wfs = [wf_dict[wid] for wid in sorted(test_ids)]

# Build vocab on train only
train_texts = []
for w in train_wfs:
    train_texts.append(w['text_repr'])
    train_texts.extend(w['queries'])

tokenizer = CustomTokenizer(min_freq=1)
vocab_size = tokenizer.build_vocab(train_texts)

# Dataset
class RetrievalDataset(Dataset):
    def __init__(self, workflows, tokenizer, max_query_len=32, max_wf_len=128):
        self.samples = []
        for w in workflows:
            wf_text = w['text_repr']
            wf_id = w['workflow_id']
            for q in w['queries']:
                self.samples.append((q, wf_text, wf_id))
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_wf_len = max_wf_len
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        query, wf_text, wf_id = self.samples[idx]
        q_ids, q_mask = self.tokenizer.encode(query, max_len=self.max_query_len)
        w_ids, w_mask = self.tokenizer.encode(wf_text, max_len=self.max_wf_len)
        return (
            torch.tensor(q_ids, dtype=torch.long),
            torch.tensor(q_mask, dtype=torch.float),
            torch.tensor(w_ids, dtype=torch.long),
            torch.tensor(w_mask, dtype=torch.float),
            wf_id
        )

train_dataset = RetrievalDataset(train_wfs, tokenizer)
val_dataset = RetrievalDataset(val_wfs, tokenizer)
test_dataset = RetrievalDataset(test_wfs, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

# Architecture
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, projection_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.att_dense = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att_vec = nn.Linear(hidden_dim, 1, bias=False)
        self.projection = nn.Linear(hidden_dim * 2, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids) # (B, L, E)
        embedded = self.dropout(embedded)
        
        lstm_out, _ = self.lstm(embedded) # (B, L, 2*H)
        lstm_out = self.dropout(lstm_out)
        
        u = torch.tanh(self.att_dense(lstm_out)) # (B, L, H)
        att_score = self.att_vec(u) # (B, L, 1)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            att_score = att_score.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(att_score, dim=1) # (B, L, 1)
        pooled = torch.sum(weights * lstm_out, dim=1) # (B, 2*H)
        
        proj = self.projection(pooled) # (B, P)
        proj = self.layer_norm(proj)
        normalized = F.normalize(proj, p=2, dim=-1) # (B, P)
        return normalized

class DualEncoderRetrievalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, projection_dim=128, temperature=0.07):
        super().__init__()
        self.query_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim, projection_dim)
        self.workflow_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim, projection_dim)
        self.temperature = temperature
        
    def forward(self, query_ids, query_mask, wf_ids, wf_mask):
        q_emb = self.query_encoder(query_ids, query_mask)
        w_emb = self.workflow_encoder(wf_ids, wf_mask)
        return q_emb, w_emb

model = DualEncoderRetrievalModel(vocab_size).to(device)

def compute_infonce_loss(q_emb, w_emb, temperature=0.07):
    sim_matrix = torch.matmul(q_emb, w_emb.T) / temperature # (B, B)
    batch_size = q_emb.size(0)
    labels = torch.arange(batch_size, device=q_emb.device)
    loss_q2w = F.cross_entropy(sim_matrix, labels)
    loss_w2q = F.cross_entropy(sim_matrix.T, labels)
    return (loss_q2w + loss_w2q) / 2.0

# Metrics
def compute_retrieval_metrics(ranks, k_list=[1, 5]):
    recalls = {k: np.mean([1.0 if r <= k else 0.0 for r in ranks]) for k in k_list}
    mrr = np.mean([1.0 / r for r in ranks])
    ndcg5 = np.mean([1.0 / math.log2(r + 1) if r <= 5 else 0.0 for r in ranks])
    return {
        "Recall@1": recalls[1],
        "Recall@5": recalls[5],
        "MRR": mrr,
        "nDCG@5": ndcg5
    }

def evaluate_retrieval(model, candidate_wfs, eval_dataset, tokenizer, device, max_wf_len=128):
    model.eval()
    cand_ids = [w['workflow_id'] for w in candidate_wfs]
    cand_texts = [w['text_repr'] for w in candidate_wfs]
    
    # Pre-encode all candidate workflows
    encoded_cand = [tokenizer.encode(t, max_len=max_wf_len) for t in cand_texts]
    cand_input_ids = torch.tensor([x[0] for x in encoded_cand], dtype=torch.long, device=device)
    cand_masks = torch.tensor([x[1] for x in encoded_cand], dtype=torch.float, device=device)
    
    with torch.no_grad():
        wf_embeddings = model.workflow_encoder(cand_input_ids, cand_masks) # (N_cand, D)
        
    ranks = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for q_ids, q_mask, _, _, target_id in DataLoader(eval_dataset, batch_size=64):
            q_ids = q_ids.to(device)
            q_mask = q_mask.to(device)
            q_embs = model.query_encoder(q_ids, q_mask) # (B_q, D)
            sims = torch.matmul(q_embs, wf_embeddings.T).cpu().numpy() # (B_q, N_cand)
            
            for i in range(len(target_id)):
                tgt = target_id[i]
                sim_row = sims[i]
                sorted_indices = np.argsort(-sim_row)
                ranked_wfs = [cand_ids[idx] for idx in sorted_indices]
                rank = ranked_wfs.index(tgt) + 1
                ranks.append(rank)
    latency_ms = ((time.perf_counter() - t0) / len(eval_dataset)) * 1000.0
    metrics = compute_retrieval_metrics(ranks)
    metrics["Latency (ms)"] = latency_ms
    return metrics

# Test Untrained Random Model
print("\n--- Evaluating Random Untrained Model ---")
rand_metrics = evaluate_retrieval(model, test_wfs, test_dataset, tokenizer, device)
for k, v in rand_metrics.items():
    print(f"{k}: {v:.4f}")

# Quick training run for 10 epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("\n--- Starting Training (10 epochs test) ---")
for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    for q_ids, q_mask, w_ids, w_mask, _ in train_loader:
        q_ids, q_mask = q_ids.to(device), q_mask.to(device)
        w_ids, w_mask = w_ids.to(device), w_mask.to(device)
        
        optimizer.zero_grad()
        q_emb, w_emb = model(q_ids, q_mask, w_ids, w_mask)
        loss = compute_infonce_loss(q_emb, w_emb, model.temperature)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    val_metrics = evaluate_retrieval(model, val_wfs, val_dataset, tokenizer, device)
    print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val R@1: {val_metrics['Recall@1']:.4f} | Val MRR: {val_metrics['MRR']:.4f}")

print("\n--- Evaluating Trained Neural Model on Test Set ---")
test_metrics = evaluate_retrieval(model, test_wfs, test_dataset, tokenizer, device)
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
