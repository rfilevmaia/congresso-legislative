# Congresso NLP — Votações Parlamentares Brasileiras

Sistema modular para coleta, enriquecimento e análise de votações da Câmara dos Deputados.
Projetado para execução em **Apple Silicon** com modelos NLP locais e persistência em **PostgreSQL (AWS RDS)**.

---

## Arquitetura

```
congresso_nlp/
├── agents/
│   ├── collector_agent.py      # Coleta votações e votos via API da Câmara
│   ├── nlp_agent.py            # Resumo + palavras-chave (modelos locais)
│   └── pipeline_agent.py      # Orquestrador: executa coleta → NLP → persistência
├── core/
│   ├── api_camara.py           # Funções puras da API da Câmara dos Deputados
│   ├── nlp_local.py            # BERTimbau (keywords) + Ollama/MLX (resumo)
│   └── database.py             # Conexão e operações PostgreSQL (SQLAlchemy)
├── models/
│   └── schema.py               # Modelos ORM (SQLAlchemy)
├── config.py                   # Configurações e variáveis de ambiente
├── requirements.txt
└── README.md
```

---

## Stack de Modelos NLP Locais (Apple Silicon)

| Tarefa | Modelo recomendado | Runtime | RAM mínima |
|---|---|---|---|
| **Keywords** | `neuralmind/bert-large-portuguese-cased` (BERTimbau Large) | 🤗 Transformers + MPS | 4 GB |
| **Keywords alternativo** | `DeBERTinha` (DeBERTa v3 XSmall PT-BR) | 🤗 Transformers + MPS | 2 GB |
| **Resumo (sumarização)** | `Qwen2.5-7B-Instruct` | Ollama (MLX backend) | 8 GB |
| **Resumo robusto** | `Qwen2.5-14B-Instruct` | Ollama (MLX backend) | 16 GB |

> **Por quê Qwen2.5 para resumo?** Benchmarks em Apple Silicon (M2 Ultra) mostram que o Qwen2.5-7B com Ollama/MLX atinge ~150 tok/s e tem excelente qualidade em português, superando Llama 3.1 8B no mesmo hardware. BERTimbau Large é o estado-da-arte para extração de features em PT-BR.

---

## Banco de Dados AWS — Recomendação de Menor Custo

### ✅ Opção recomendada: **Amazon RDS PostgreSQL** (db.t4g.micro)

| Opção | Custo estimado/mês | Observação |
|---|---|---|
| **RDS PostgreSQL db.t4g.micro** | ~**$15–20/mês** | Melhor custo-benefício para uso moderado |
| RDS PostgreSQL db.t3.micro | ~$15–18/mês | Alternativa sem Graviton |
| Aurora Serverless v2 | ~$40–70/mês | Mais caro; vale apenas para cargas imprevisíveis |
| Aurora Provisioned t3.medium | ~$69+/mês | Superdimensionado para este caso |

> **RDS PostgreSQL db.t4g.micro** é a escolha ideal: free tier disponível por 12 meses (750h/mês), Graviton2 para melhor price-performance, suporte nativo a JSONB (ideal para keywords e metadados), extensão `pg_trgm` para busca textual, e custo fixo previsível.

---

## Instalação

```bash
# 1. Dependências Python
pip install -r requirements.txt

# 2. Modelos locais — instalar Ollama (macOS)
brew install ollama
ollama pull qwen2.5:7b          # modelo de resumo padrão
# ollama pull qwen2.5:14b       # versão mais robusta (requer 16 GB RAM)

# 3. Variáveis de ambiente
cp .env.example .env
# editar .env com credenciais do RDS

# 4. Criar tabelas no banco
python -c "from models.schema import init_db; init_db()"
```

---

## Uso Rápido

```python
from agents.pipeline_agent import PipelineAgent

agent = PipelineAgent()

# Coletar votações de um período e enriquecer com NLP
agent.run(data_inicio="2024-03-01", data_fim="2024-03-31")

# Consultar como um deputado votou em temas específicos
from core.database import query_deputado_votacoes
rows = query_deputado_votacoes(nome_deputado="Tabata Amaral", tema_keyword="educação")
```
