import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Let's test TF-IDF and Reranking logic
from test_training import (
    train_wfs, test_wfs, test_dataset, tokenizer, model, device,
    compute_retrieval_metrics
)

# 1. TF-IDF Baseline
print("\n--- Evaluating TF-IDF Baseline ---")
train_corpus = [w['text_repr'] for w in train_wfs]
tfidf_vec = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf_vec.fit(train_corpus)

test_cand_ids = [w['workflow_id'] for w in test_wfs]
test_wf_matrix = tfidf_vec.transform([w['text_repr'] for w in test_wfs])

tfidf_ranks = []
for sample in test_dataset.samples:
    query, _, target_id = sample
    q_vec = tfidf_vec.transform([query])
    sims = sklearn_cosine_similarity(q_vec, test_wf_matrix)[0]
    sorted_idx = np.argsort(-sims)
    ranked_ids = [test_cand_ids[idx] for idx in sorted_idx]
    rank = ranked_ids.index(target_id) + 1
    tfidf_ranks.append(rank)

tfidf_metrics = compute_retrieval_metrics(tfidf_ranks)
for k, v in tfidf_metrics.items():
    print(f"TF-IDF {k}: {v:.4f}")

# 2. Test Reranking on Test Set
print("\n--- Testing Reranker Variants ---")

def compute_structured_scores(wf):
    # Temporal: exponential decay
    days_ago = wf.get('days_ago', 15.0)
    s_temp = math.exp(-0.05 * days_ago)
    
    # Status
    st = wf.get('status', 'COMPLETED')
    if st == 'FAILED':
        s_stat = 1.0
    elif st == 'INCOMPLETE':
        s_stat = 0.75
    else:
        s_stat = 0.30
        
    # Dependency
    nodes = wf.get('nodes', [])
    num_nodes = max(len(nodes), 1)
    num_inc = sum(1 for n in nodes if n['status'] == 'INCOMPLETE')
    num_fail = sum(1 for n in nodes if n['status'] == 'FAILED')
    s_dep = min(1.0, (num_inc + 1.5 * num_fail) / num_nodes)
    
    return s_temp, s_stat, s_dep

import torch
# Precompute test candidate workflow embeddings
cand_texts = [w['text_repr'] for w in test_wfs]
cand_ids = [w['workflow_id'] for w in test_wfs]
encoded_cand = [tokenizer.encode(t, max_len=128) for t in cand_texts]
cand_input_ids = torch.tensor([x[0] for x in encoded_cand], dtype=torch.long, device=device)
cand_masks = torch.tensor([x[1] for x in encoded_cand], dtype=torch.float, device=device)

with torch.no_grad():
    wf_embeddings = model.workflow_encoder(cand_input_ids, cand_masks)

# Evaluate each ablation
variants = {
    "Custom Neural (Semantic Only)": (1.0, 0.0, 0.0, 0.0),
    "Custom + Status": (0.80, 0.0, 0.20, 0.0),
    "Custom + Status + Temporal": (0.70, 0.10, 0.20, 0.0),
    "Full Model (Semantic+Temp+Stat+Dep)": (0.60, 0.10, 0.20, 0.10)
}

wf_features = {w['workflow_id']: compute_structured_scores(w) for w in test_wfs}

for var_name, (w_sem, w_temp, w_stat, w_dep) in variants.items():
    ranks = []
    for sample in test_dataset.samples:
        query, _, target_id = sample
        q_ids, q_mask = tokenizer.encode(query, max_len=32)
        with torch.no_grad():
            q_emb = model.query_encoder(
                torch.tensor([q_ids], dtype=torch.long, device=device),
                torch.tensor([q_mask], dtype=torch.float, device=device)
            )
            sims = torch.matmul(q_emb, wf_embeddings.T).cpu().numpy()[0]
            
        scores = []
        for idx, wid in enumerate(cand_ids):
            raw_sim = float(sims[idx])
            s_sem = max(0.0, (raw_sim + 1.0) / 2.0)
            s_temp, s_stat, s_dep_score = wf_features[wid]
            
            # Semantic gating
            gate = min(1.0, max(0.0, (s_sem - 0.40) / 0.30)) if w_sem < 1.0 else 0.0
            
            final_score = (
                w_sem * s_sem +
                gate * (w_temp * s_temp + w_stat * s_stat + w_dep * s_dep_score)
            )
            scores.append(final_score)
            
        sorted_idx = np.argsort(-np.array(scores))
        ranked_ids = [cand_ids[idx] for idx in sorted_idx]
        rank = ranked_ids.index(target_id) + 1
        ranks.append(rank)
        
    met = compute_retrieval_metrics(ranks)
    print(f"\n{var_name}:")
    for k, v in met.items():
        print(f"  {k}: {v:.4f}")
