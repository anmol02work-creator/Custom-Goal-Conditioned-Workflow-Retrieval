"""
predictor.py - Production Inference Engine for Model A: Goal-Conditioned Workflow Retrieval
Model Version: JARVIS-WorkflowRetrieval-v2.0
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_VERSION = "JARVIS-WorkflowRetrieval-v2.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

class WorkflowPredictor:
    def __init__(self, model_dir=MODEL_DIR, tokenizer_dir=TOKENIZER_DIR, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        
    def _encode_texts(self, texts, max_length=128):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_emb = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_emb / sum_mask
            normalized = F.normalize(pooled, p=2, dim=1)
        return normalized

    def retrieve(self, query: str, current_state: dict = None, candidate_workflows: list = None, top_k: int = 5) -> dict:
        if not candidate_workflows:
            return {"query": query, "results": [], "model_version": MODEL_VERSION}
            
        N = len(candidate_workflows)
        c_texts = []
        for cand in candidate_workflows:
            goal = cand.get("goal", "").strip()
            domain = cand.get("domain", "").strip()
            status = cand.get("overall_status", "INCOMPLETE")
            nodes = cand.get("graph", {}).get("nodes", [])
            act_str = " -> ".join([f"{n.get('node_id', idx+1)}. {n.get('description', '')}" for idx, n in enumerate(nodes)])
            c_texts.append(f"Domain: {domain} | Goal: {goal} | Status: {status} | Actions: {act_str}")
            
        q_emb = self._encode_texts([query], max_length=64)
        c_embs = self._encode_texts(c_texts, max_length=128)
        
        sims = torch.matmul(q_emb, c_embs.T)[0].cpu().numpy()
        sem_scores = (sims + 1.0) / 2.0
        
        # Status boosting
        q_lower = query.lower()
        is_cont = any(k in q_lower for k in ["continue", "resume", "finish", "pick up", "fix", "debug", "error", "failed", "leave off"])
        status_scores = np.zeros(N)
        for i, cand in enumerate(candidate_workflows):
            st = cand.get("overall_status", "INCOMPLETE")
            if is_cont:
                if st == "FAILED": status_scores[i] = 0.20
                elif st == "INCOMPLETE": status_scores[i] = 0.15
                else: status_scores[i] = -0.10
            else:
                if st == "FAILED": status_scores[i] = 0.10
                elif st == "INCOMPLETE": status_scores[i] = 0.05
                else: status_scores[i] = 0.00
                
        # Context overlap
        ctx_scores = np.zeros(N)
        if current_state:
            curr_apps = set(a.lower() for a in current_state.get("applications", []))
            curr_files = set(f.lower() for f in current_state.get("files", []))
            for i, cand in enumerate(candidate_workflows):
                c_st = cand.get("current_state", {})
                w_apps = set(a.lower() for a in c_st.get("applications", []))
                w_files = set(f.lower() for f in c_st.get("files", []))
                app_m = len(curr_apps.intersection(w_apps)) / max(len(curr_apps), 1)
                file_m = len(curr_files.intersection(w_files)) / max(len(curr_files), 1)
                ctx_scores[i] = 0.6 * app_m + 0.4 * file_m
                
        final_scores = 0.60 * sem_scores + 0.25 * status_scores + 0.15 * ctx_scores
        final_scores = np.clip(final_scores, 0.0, 1.0)
        ranked_indices = np.argsort(final_scores)[::-1]
        
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            cand = candidate_workflows[idx]
            nodes = cand.get("graph", {}).get("nodes", [])
            st = cand.get("overall_status", "INCOMPLETE")
            if st == "COMPLETED":
                ret_nodes = [n.get("node_id", j+1) for j, n in enumerate(nodes)]
            else:
                ret_nodes = [n.get("node_id", j+1) for j, n in enumerate(nodes) if n.get("status") in ["FAILED", "INCOMPLETE"]]
                if not ret_nodes:
                    ret_nodes = [n.get("node_id", j+1) for j, n in enumerate(nodes)]
                    
            results.append({
                "workflow_id": cand["workflow_id"],
                "score": float(np.round(final_scores[idx], 2)),
                "retrieved_node_ids": ret_nodes
            })
            
        return {
            "query": query,
            "results": results,
            "model_version": MODEL_VERSION
        }

_predictor = None

def retrieve_workflows(query: str, current_state: dict = None, candidate_workflows: list = None, top_k: int = 5) -> dict:
    global _predictor
    if _predictor is None:
        _predictor = WorkflowPredictor()
    return _predictor.retrieve(query, current_state, candidate_workflows, top_k)
