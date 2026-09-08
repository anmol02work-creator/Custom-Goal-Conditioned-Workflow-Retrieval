# Model A: Goal-Conditioned Workflow Retrieval Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Version](https://img.shields.io/badge/Model%20Version-JARVIS--WorkflowRetrieval--v2.0-orange.svg)]()

A complete, research-grade, independent model for **Goal-Conditioned Workflow Retrieval and Subgraph Extraction**. Given a natural language user goal, active desktop application context, and historical workflow memories, Model A retrieves the most relevant workflow and extracts the actionable continuation subgraph (`retrieved_node_ids`).

---

## 🚀 Key Highlights

1. **Exact Contracts Adherence**:
   - **Inference Input**: Accepts `query`, `current_state` (`applications`, `files`, `browser_tabs`), and `candidate_workflows` containing `workflow_id`, `goal`, `graph` (DAG nodes & dependencies), and `checkpoint`.
   - **Inference Output**: Returns `workflow_id`, calibrated `score`, and actionable `retrieved_node_ids` (the connected continuation subgraph).
2. **Dual-Track Research Benchmark**:
   - **Real-World Empirical Track**: Validated on real human web navigation trajectories from the peer-reviewed **Mind2Web** benchmark (Deng et al., NeurIPS 2023), preserving original dataset UUIDs (`annotation_id`, `action_uid`).
   - **Large-Scale Desktop DAG Track**: Evaluated on **1,000 multi-domain desktop workflows** across 10 technical domains with 5,000 user queries, DAG execution dependencies, checkpoints across 30 days, and desktop application states.
3. **Contrastive Fine-Tuning**:
   - Fine-tuned `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters) using symmetric **InfoNCE / Multiple Negatives Ranking Loss (MNRL)** ($\tau = 0.05$) with in-batch negatives.
   - Strict leakage prevention: Partitioned strictly by unique workflow IDs (700 Train, 150 Val, 150 Test) with zero intersection.
4. **Multi-Factor Ranking Formulation**:
   $$S_{\text{final}} = 0.55 \cdot S_{\text{semantic}} + 0.15 \cdot R_{\text{temporal}} + 0.20 \cdot B_{\text{status}} + 0.10 \cdot S_{\text{context}}$$
   - **Semantic Similarity ($S_{\text{semantic}}$)**: Cosine similarity via fine-tuned Sentence-BERT.
   - **Temporal Decay ($R_{\text{temporal}}$)**: Exponential recency decay $\exp(-0.05 \cdot \Delta t_{\text{days}})$ based on checkpoint timestamp.
   - **Status Priority ($B_{\text{status}}$)**: Urgency boost (`FAILED` > `INCOMPLETE` > `COMPLETED`) when users request continuation.
   - **Desktop Context ($S_{\text{context}}$)**: Jaccard overlap with active desktop tools, open files, and browser tabs.
5. **Connected Subgraph Extraction**:
   - Traverses DAG topological dependencies to extract the active frontier of uncompleted/failed nodes whose prerequisites are met, plus their downstream continuation subgraph, ensuring autonomous agents do not re-run completed steps.

---

## 📊 Four-Stage Benchmark Progression Results

Evaluated on the untouched test partition (150 test workflows, 750 query formulations):

| Progression Stage | Recall@1 | Recall@5 | MRR | nDCG@5 | Latency (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: TF-IDF Lexical Baseline** | 0.5627 | 0.8973 | 0.7111 | 0.7506 | 0.86 ms |
| **Stage 2: SBERT Zero-Shot Semantic** | 0.5800 | 0.8973 | 0.7211 | 0.7597 | 15.59 ms |
| **Stage 3: Status-Aware Semantic** | 0.7187 | 0.9653 | 0.8253 | 0.8571 | 14.56 ms |
| **Stage 4: Full Multi-Factor Retrieval** | **0.8720** | **0.9947** | **0.9259** | **0.9428** | **15.10 ms** |

---

## 💻 Exact Public Python API

```python
from predictor import retrieve_workflows

candidate_workflows = [
    {
        "workflow_id": "workflow_017",
        "goal": "Build a Python project",
        "domain": "Python development",
        "overall_status": "FAILED",
        "days_ago": 1.2,
        "graph": {
            "nodes": [
                {"node_id": 1, "description": "Create project directory", "status": "COMPLETED"},
                {"node_id": 2, "description": "Set up virtualenv", "status": "COMPLETED"},
                {"node_id": 3, "description": "Implement core logic", "status": "COMPLETED"},
                {"node_id": 4, "description": "Run unit test suite", "status": "FAILED"},
                {"node_id": 5, "description": "Configure linter", "status": "INCOMPLETE"},
                {"node_id": 7, "description": "Build Docker container", "status": "INCOMPLETE"},
                {"node_id": 8, "description": "Deploy to staging", "status": "INCOMPLETE"}
            ],
            "dependencies": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 7], [7, 8]]
        },
        "checkpoint": {
            "timestamp": "2026-09-05T10:30:00",
            "last_active_node": 4,
            "state_summary": "Test suite failed on assertion error in test_core.py"
        },
        "current_state": {
            "applications": ["Visual Studio Code"],
            "files": ["main.py"],
            "browser_tabs": []
        }
    }
]

response = retrieve_workflows(
    query="Continue my previous coding work",
    current_state={
        "applications": ["Visual Studio Code"],
        "files": ["main.py"],
        "browser_tabs": []
    },
    candidate_workflows=candidate_workflows,
    top_k=5
)
print(response)
```

### Response
```json
{
  "query": "Continue my previous coding work",
  "results": [
    {
      "workflow_id": "workflow_017",
      "score": 0.94,
      "retrieved_node_ids": [4, 5, 7, 8]
    }
  ],
  "model_version": "JARVIS-WorkflowRetrieval-v2.0"
}
```

---

## 📁 Repository Structure

```
├── JARVIS_Model_A_Workflow_Retrieval.ipynb  # Self-contained Google Colab notebook
├── build_notebook.py                        # Notebook generation script
├── generate_large_scale_dataset.py          # 1,000-workflow benchmark generator
├── large_scale_workflows.json               # 1,000 workflow dataset (2.59 MB)
├── extract_real_data.py                     # Mind2Web streaming extraction script
├── real_mind2web_tasks.json                 # Clean Mind2Web real benchmark dataset
├── dataset_audit.py / .json                 # Mind2Web and OSWorld audit report
├── dataset_provenance.json                  # Dataset provenance metadata
├── train_and_evaluate_v2.py                 # Multi-factor training & 4-stage progression
├── jarvis_workflow_retrieval_model/         # Packaged model weights, config & predictor
│   ├── model/                               # PyTorch transformer weights
│   ├── tokenizer/                           # Tokenizer configs and vocabulary
│   └── predictor.py                         # Standalone inference engine
└── README.md                                # Project documentation
```

---

## 🛠️ Quickstart & Reproduction

### Option 1: Run in Google Colab
1. Upload `JARVIS_Model_A_Workflow_Retrieval.ipynb` to Google Colab.
2. Select **Runtime $\to$ Run all**.
3. The notebook executes data preparation, contrastive training, 4-stage evaluation, and launches the interactive `ipywidgets` GUI in ~2 minutes.

### Option 2: Run Locally
```bash
# 1. Install dependencies
pip install torch transformers sentence-transformers scikit-learn ipywidgets matplotlib seaborn

# 2. Run training and evaluation pipeline
python train_and_evaluate_v2.py
```

---

## 📜 License
Apache 2.0 License.
