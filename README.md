# Congresso NLP — Brazilian Parliamentary Votings

Modular system for collecting, enriching and analysing votings from the Chamber of Deputies.
Designed to run on **Apple Silicon** with local NLP models and persistence in **PostgreSQL (AWS RDS)**.

---

## Architecture

```
congresso_nlp/
├── agents/
│   ├── collector_agent.py      # Collects votings and votes via the Câmara API
│   ├── nlp_agent.py            # Summary + keywords (local models)
│   └── pipeline_agent.py      # Orchestrator: runs collection → NLP → persistence
├── core/
│   ├── api_camara.py           # Pure functions for the Câmara dos Deputados API
│   ├── nlp_local.py            # BERTimbau (keywords) + Ollama/MLX (summary)
│   └── database.py             # PostgreSQL connection and operations (SQLAlchemy)
├── models/
│   └── schema.py               # ORM models (SQLAlchemy)
├── config.py                   # Configuration and environment variables
├── requirements.txt
└── README.md
```

---

## Local NLP Model Stack (Apple Silicon)

| Task | Recommended model | Runtime | Minimum RAM |
|---|---|---|---|
| **Keywords** | `neuralmind/bert-large-portuguese-cased` (BERTimbau Large) | 🤗 Transformers + MPS | 4 GB |
| **Keywords (alternative)** | `DeBERTinha` (DeBERTa v3 XSmall PT-BR) | 🤗 Transformers + MPS | 2 GB |
| **Summary** | `Qwen2.5-7B-Instruct` | Ollama (MLX backend) | 8 GB |
| **Robust summary** | `Qwen2.5-14B-Instruct` | Ollama (MLX backend) | 16 GB |

> **Why Qwen2.5 for summaries?** Benchmarks on Apple Silicon (M2 Ultra) show that Qwen2.5-7B with Ollama/MLX achieves ~150 tok/s and produces excellent quality output in Portuguese, outperforming Llama 3.1 8B on the same hardware. BERTimbau Large is the state of the art for feature extraction in Brazilian Portuguese.

---

## AWS Database — Lowest Cost Recommendation

### ✅ Recommended option: **Amazon RDS PostgreSQL** (db.t4g.micro)

| Option | Estimated cost/month | Notes |
|---|---|---|
| **RDS PostgreSQL db.t4g.micro** | ~**$15–20/month** | Best cost-to-performance ratio for moderate usage |
| RDS PostgreSQL db.t3.micro | ~$15–18/month | Alternative without Graviton |
| Aurora Serverless v2 | ~$40–70/month | More expensive; only worthwhile for unpredictable workloads |
| Aurora Provisioned t3.medium | ~$69+/month | Oversized for this use case |

> **RDS PostgreSQL db.t4g.micro** is the ideal choice: free tier available for 12 months (750 hours/month), Graviton2 for better price-performance, native JSONB support (ideal for keywords and metadata), `pg_trgm` extension for full-text search, and predictable fixed costs.

---

## Installation

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. Local models — install Ollama (macOS)
brew install ollama
ollama pull qwen2.5:7b          # default summary model
# ollama pull qwen2.5:14b       # more robust version (requires 16 GB RAM)

# 3. Environment variables
cp .env.example .env
# edit .env with your RDS credentials

# 4. Create database tables
python -c "from models.schema import init_db; init_db()"
```

---

## Quick Start

```python
from agents.pipeline_agent import PipelineAgent

agent = PipelineAgent()

# Collect votings for a period and enrich with NLP
agent.run(data_inicio="2024-03-01", data_fim="2024-03-31")

# Query how a deputy voted on specific topics
from core.database import query_deputado_votacoes
rows = query_deputado_votacoes(nome_deputado="Tabata Amaral", tema_keyword="educação")
```
