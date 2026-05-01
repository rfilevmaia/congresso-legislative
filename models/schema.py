"""
models/schema.py — Modelos ORM (SQLAlchemy 2.x).

Tabelas:
  deputados         → cadastro do parlamentar
  votacoes          → cada votação (proposição + resultado)
  votos             → voto de cada deputado em cada votação
  proposicoes       → proposição legislativa associada à votação
  votacao_nlp       → resumo e keywords gerados pelo pipeline NLP
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean,
    DateTime, Text, ForeignKey, UniqueConstraint, Index, Float
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship, Session
from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


# ── Deputados ─────────────────────────────────────────────────────────────────

class Deputado(Base):
    __tablename__ = "deputados"

    id              = Column(Integer, primary_key=True)       # id oficial da API
    nome_civil      = Column(String(200), nullable=False)
    nome_eleitoral  = Column(String(200))
    partido         = Column(String(20))
    uf              = Column(String(2))
    legislatura     = Column(Integer)
    uri             = Column(String(300))
    foto_url        = Column(String(300))
    criado_em       = Column(DateTime, default=datetime.utcnow)
    atualizado_em   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    votos           = relationship("Voto", back_populates="deputado")
    discursos       = relationship("Discurso", back_populates="deputado")

    def __repr__(self):
        return f"<Deputado {self.id} — {self.nome_eleitoral} ({self.partido}/{self.uf})>"


# ── Proposições ───────────────────────────────────────────────────────────────

class Proposicao(Base):
    __tablename__ = "proposicoes"

    id              = Column(Integer, primary_key=True)
    uri             = Column(String(300))
    sigle_tipo      = Column(String(20))       # PL, PEC, MPV, etc.
    numero          = Column(Integer)
    ano             = Column(Integer)
    ementa          = Column(Text)             # descrição oficial
    keywords_api    = Column(JSONB)            # keywords vindas da própria API
    temas_api       = Column(JSONB)            # temas temáticos da API
    url_inteiro_teor = Column(String(300))
    criado_em       = Column(DateTime, default=datetime.utcnow)

    votacoes        = relationship("Votacao", back_populates="proposicao")


# ── Votações ──────────────────────────────────────────────────────────────────

class Votacao(Base):
    __tablename__ = "votacoes"

    id              = Column(String(100), primary_key=True)   # id da API (string)
    uri             = Column(String(300))
    data            = Column(DateTime)
    descricao       = Column(Text)             # descrição da votação
    aprovacao       = Column(Boolean)          # True = aprovado, False = rejeitado
    placar_sim      = Column(Integer)
    placar_nao      = Column(Integer)
    placar_abstencao = Column(Integer)
    id_orgao        = Column(String(50))
    id_evento       = Column(String(50))

    proposicao_id   = Column(Integer, ForeignKey("proposicoes.id"), nullable=True)
    proposicao      = relationship("Proposicao", back_populates="votacoes")
    votos           = relationship("Voto", back_populates="votacao")
    nlp             = relationship("VotacaoNLP", back_populates="votacao", uselist=False)

    criado_em       = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        status = "✅ APROVADO" if self.aprovacao else "❌ REJEITADO"
        return f"<Votacao {self.id} — {status} em {self.data}>"


# ── Votos ─────────────────────────────────────────────────────────────────────

class Voto(Base):
    __tablename__ = "votos"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    votacao_id      = Column(String(100), ForeignKey("votacoes.id"), nullable=False)
    deputado_id     = Column(Integer, ForeignKey("deputados.id"), nullable=False)

    # Valores possíveis: "Sim", "Não", "Abstenção", "Obstrução", "Art. 17", "Presidente"
    voto            = Column(String(30), nullable=False)
    hora_registro   = Column(DateTime)
    criado_em       = Column(DateTime, default=datetime.utcnow)

    votacao         = relationship("Votacao", back_populates="votos")
    deputado        = relationship("Deputado", back_populates="votos")

    __table_args__ = (
        UniqueConstraint("votacao_id", "deputado_id", name="uq_voto_votacao_deputado"),
        Index("ix_votos_deputado_id", "deputado_id"),
        Index("ix_votos_votacao_id", "votacao_id"),
    )


# ── NLP por Votação ───────────────────────────────────────────────────────────

class VotacaoNLP(Base):
    __tablename__ = "votacoes_nlp"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    votacao_id      = Column(String(100), ForeignKey("votacoes.id"), unique=True, nullable=False)

    # Gerados pelo pipeline NLP local
    resumo          = Column(Text)             # parágrafo de resumo (Qwen2.5 via Ollama)
    keywords        = Column(JSONB)            # lista de strings (BERTimbau / KeyBERT)
    tema_inferido   = Column(String(200))      # tema livre inferido pelo LLM
    sentimento_tema = Column(String(50))       # ex: "conservador", "progressista", "neutro"

    modelo_resumo   = Column(String(100))      # ex: "qwen3.5:9b"
    modelo_keywords = Column(String(100))      # ex: "neuralmind/bert-large-portuguese-cased"
    processado_em   = Column(DateTime, default=datetime.utcnow)

    # Status do enriquecimento — controla o que o EnrichmentAgent deve fazer
    # Valores possíveis:
    #   None          → ainda não tentou enriquecer
    #   "pendente"    → tem votos, aguarda enriquecimento
    #   "enriquecido" → texto integral encontrado e processado
    #   "sem_texto"   → tentou todas as fontes, nenhum texto disponível
    #   "sem_votos"   → votação sem votos nominais — ignorar no enrichment
    status_enriquecimento = Column(String(30), nullable=True, index=True)

    votacao         = relationship("Votacao", back_populates="nlp")



# ── Discursos (importados do parquet de análise de sentimentos) ───────────────

class Discurso(Base):
    __tablename__ = "discursos"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    id_deputado     = Column(Integer, ForeignKey("deputados.id"), nullable=False)
    data            = Column(DateTime, nullable=False)
    keywords_api    = Column(JSONB)          # keywords da API da Câmara (coluna keywords)
    keywords_tfidf  = Column(JSONB)          # keywords TF-IDF do texto (coluna tfidf_keywords)
    transcricao     = Column(Text)           # texto original
    transcricao_limpa = Column(Text)         # coluna transcription_clean
    local           = Column(String(100))
    label           = Column(String(10))     # POS / NEG / NEU
    label_emotion   = Column(String(50))     # others, joy, fear, etc.
    confidence      = Column(Float)
    resumo          = Column(Text)           # gerado pelo Qwen3.5 local (pode ser nulo)
    resumo_gerado   = Column(Boolean, default=False)  # flag: resumo já foi gerado?
    criado_em       = Column(DateTime, default=datetime.utcnow)

    deputado        = relationship("Deputado", back_populates="discursos")
    comparacoes     = relationship("DiscursoVotacaoComparacao", back_populates="discurso")

    __table_args__ = (
        Index("ix_discursos_deputado_data", "id_deputado", "data"),
    )


# ── Comparações discurso × votação ────────────────────────────────────────────

class DiscursoVotacaoComparacao(Base):
    __tablename__ = "discurso_votacao_comparacoes"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    discurso_id     = Column(Integer, ForeignKey("discursos.id"), nullable=False)
    votacao_id      = Column(String(100), ForeignKey("votacoes.id"), nullable=False)
    id_deputado     = Column(Integer, ForeignKey("deputados.id"), nullable=False)

    # Métricas de sobreposição de keywords
    keywords_comuns     = Column(JSONB)      # lista de keywords em comum
    n_keywords_comuns   = Column(Integer, default=0)
    score_overlap       = Column(Float)      # jaccard: |intersecao| / |uniao|

    # Voto e sentimento
    voto                = Column(String(30)) # Sim / Nao / Abstencao / ...
    sentimento_discurso = Column(String(10)) # POS / NEG / NEU
    confidence          = Column(Float)

    # Coerência: discurso × voto
    # "coerente"   = discurso positivo sobre o tema + voto Sim (ou negativo + Nao)
    # "incoerente" = discurso positivo + voto Nao (ou negativo + Sim)
    # "neutro"     = abstencao, ou sentimento NEU, ou overlap baixo
    coerencia           = Column(String(20)) # coerente / incoerente / neutro / indefinido
    score_coerencia     = Column(Float)      # -1.0 (muito incoerente) a +1.0 (muito coerente)

    # Janela temporal usada no match
    dias_diferenca      = Column(Integer)    # discurso - votacao em dias (negativo = antes)

    criado_em           = Column(DateTime, default=datetime.utcnow)

    discurso    = relationship("Discurso", back_populates="comparacoes")
    votacao     = relationship("Votacao")
    deputado    = relationship("Deputado")

    __table_args__ = (
        UniqueConstraint("discurso_id", "votacao_id", name="uq_comparacao_discurso_votacao"),
        Index("ix_comparacoes_deputado", "id_deputado"),
        Index("ix_comparacoes_coerencia", "coerencia"),
    )

# ── Utilitários ───────────────────────────────────────────────────────────────
# Engine e SessionFactory criados uma unica vez (singleton).
# Recriar o engine a cada chamada desperdicaria conexoes do pool.

from sqlalchemy.orm import sessionmaker as _sessionmaker

_engine = None
_SessionFactory = None


def get_engine():
    """Retorna o engine singleton, criando-o na primeira chamada."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,   # testa conexao antes de usar (detecta drops de rede)
            pool_size=5,          # conexoes persistentes no pool
            max_overflow=10,      # conexoes extras permitidas sob carga
            pool_recycle=1800,    # recicla conexoes a cada 30 min (evita timeout do PG)
        )
    return _engine


def get_session() -> Session:
    """Retorna uma nova Session usando o pool de conexoes compartilhado."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = _sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionFactory()


def init_db():
    """Cria todas as tabelas no banco (idempotente — seguro rodar multiplas vezes)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Tabelas criadas/verificadas com sucesso.")
    return engine
