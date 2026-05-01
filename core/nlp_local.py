"""
core/nlp_local.py — NLP 100% local, optimized for Apple Silicon (MPS).

Responsibilities:
  1. Keyword extraction  → BERTimbau Large via KeyBERT (MPS)
  2. Summary generation  → Qwen3.5 via Ollama
  3. Topic inference     → BERTopic (loaded from local pkl file)

Notes on Qwen3.5:
  - Has thinking mode enabled by default — requires num_predict >= 2000
  - Use /no_think suffix in prompts to disable thinking and get faster responses
  - Thinking mode consumes tokens before generating the actual response
"""

from __future__ import annotations
import re
import os
import pickle
from typing import Optional
from loguru import logger
from config import (
    KEYWORD_MODEL, KEYWORD_TOP_N, KEYWORD_DEVICE,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
)


# =============================================================================
# 1. KEYWORD EXTRACTION — BERTimbau / KeyBERT
# =============================================================================

_keybert_model = None  # lazy init — loaded on first call


def _get_keybert() -> "KeyBERT":
    """
    Initializes KeyBERT with BERTimbau Large on MPS (Apple Silicon).
    Lazy to avoid blocking imports when the module is not used.
    """
    global _keybert_model
    if _keybert_model is not None:
        return _keybert_model

    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    logger.info(f"Carregando modelo de keywords: {KEYWORD_MODEL} no device={KEYWORD_DEVICE}")
    st_model       = SentenceTransformer(KEYWORD_MODEL, device=KEYWORD_DEVICE)
    _keybert_model = KeyBERT(model=st_model)
    logger.info("✅ Modelo de keywords carregado.")
    return _keybert_model


def extrair_keywords(texto: str, top_n: int = KEYWORD_TOP_N) -> list[str]:
    """
    Extracts top-N keywords from a Portuguese text using BERTimbau.

    Algorithm: MMR (Maximal Marginal Relevance) — balances relevance and diversity.
    Supports n-grams of 1 to 3 tokens to capture compound terms.
    """
    if not texto or len(texto.strip()) < 20:
        logger.warning("Texto muito curto para extração de keywords.")
        return []

    texto = _limpar_texto(texto)

    try:
        kw_model   = _get_keybert()
        resultados = kw_model.extract_keywords(
            texto,
            keyphrase_ngram_range=(1, 3),
            stop_words=None,
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
        )
        keywords = [kw for kw, _score in resultados]
        logger.debug(f"Keywords extraídas: {keywords}")
        return keywords

    except Exception as e:
        logger.error(f"Erro na extração de keywords: {e}")
        return _fallback_keywords(texto, top_n)


def _fallback_keywords(texto: str, top_n: int) -> list[str]:
    """Statistical fallback (simple TF) when BERTimbau is unavailable."""
    STOPWORDS_PT = {
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "e", "ou", "que", "se", "ao", "um", "uma", "para", "com", "por",
        "os", "as", "a", "o", "é", "são", "foi", "ser", "ter", "não",
    }
    words = re.findall(r'\b[a-záéíóúâêîôûãõàç]{4,}\b', texto.lower())
    freq  = {}
    for w in words:
        if w not in STOPWORDS_PT:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


# =============================================================================
# 2. SUMMARY GENERATION — Qwen3.5 via Ollama
# =============================================================================

def gerar_resumo(
    texto: str,
    contexto_extra: str = "",
    modelo: str = OLLAMA_MODEL,
) -> str:
    """
    Generates a Portuguese summary of a legislative text using Qwen3.5 via Ollama.

    Uses /no_think suffix to disable Qwen3 thinking mode and get direct responses.
    num_predict set to 2000 to ensure enough tokens for thinking + response.
    """
    if not texto or len(texto.strip()) < 30:
        return "Texto insuficiente para geração de resumo."

    texto  = _limpar_texto(texto)
    prompt = _montar_prompt_resumo(texto, contexto_extra)

    try:
        import ollama
        client   = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(
            model=modelo,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "top_p":       0.9,
                "num_predict": 2000,  # high enough for qwen3 thinking + response
            },
        )
        resumo = response.get("response", "").strip()
        logger.debug(f"Resumo gerado ({len(resumo)} chars) com {modelo}")
        return resumo

    except Exception as e:
        logger.error(f"Erro ao gerar resumo com Ollama ({modelo}): {e}")
        return _fallback_resumo(texto)


def _montar_prompt_resumo(texto: str, contexto_extra: str = "") -> str:
    """
    Builds the instructional prompt for Qwen3.5.
    /no_think suffix disables thinking mode for faster, direct responses.
    """
    ctx = f"\nContexto adicional: {contexto_extra}" if contexto_extra else ""
    return f"""Você é um analista legislativo especializado no Congresso Nacional Brasileiro.
Leia o texto abaixo e escreva um resumo claro e objetivo em português brasileiro.
O resumo deve ter no máximo 3 parágrafos, destacar o tema central, o objetivo da proposta
e eventuais impactos para a população. Use linguagem acessível, sem jargões desnecessários.{ctx}

TEXTO:
{texto[:3000]}

RESUMO: /no_think"""


def _fallback_resumo(texto: str) -> str:
    """Returns the first sentences of the text as an emergency summary."""
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    return " ".join(sentencas[:3])


# =============================================================================
# 3. BERTOPIC — TOPIC INFERENCE
# =============================================================================

_bertopic_model = None

# Maps BERTopic topic IDs to legislative category strings.
# Must match the CANDIDATE_LABELS used in enrichment_agent.py.
TOPIC_TO_CATEGORY = {
    -1: "uncategorized",
     0: "uncategorized",
     1: "direitos sociais",
     2: "política externa",
     3: "direitos civis e liberdade de expressão",
     4: "pessoas com deficiência",
     5: "aborto e direito à vida",
     6: "educação",
     7: "direitos da mulher",
     8: "pandemia e saúde pública",
     9: "cultura",
    10: "educação",
    11: "crise climática",
    12: "religião",
    13: "direitos da mulher",
    14: "uncategorized",
}


def _get_bertopic() -> "BERTopic":
    """
    Loads the BERTopic model trained on political speeches.
    Uses pickle — model must have been trained on CPU to load without CUDA.
    """
    global _bertopic_model
    if _bertopic_model is not None:
        return _bertopic_model

    model_path = "/Volumes/LaCie/development/congresso/models/bertopic_v3.pkl"
    logger.info(f"Loading BERTopic from {model_path}...")

    with open(model_path, "rb") as f:
        _bertopic_model = pickle.load(f)

    logger.info("✅ BERTopic loaded.")
    return _bertopic_model


def _inferir_tema_bertopic(texto: str) -> str:
    """
    Classifies a text into a legislative topic using the trained BERTopic model.
    Falls back to keyword matching if BERTopic fails or returns low confidence.
    """
    if not texto or len(texto.strip()) < 20:
        return "uncategorized"

    CONFIDENCE_THRESHOLD = 0.30

    try:
        model      = _get_bertopic()
        topics, probs = model.transform([texto])
        topic_id   = topics[0]
        confidence = float(probs[0].max()) if hasattr(probs[0], "max") else float(probs[0])

        if confidence < CONFIDENCE_THRESHOLD:
            return _inferir_tema_fallback([], texto)

        return TOPIC_TO_CATEGORY.get(topic_id, "uncategorized")

    except Exception as e:
        logger.error(f"BERTopic.transform error: {e} — falling back to keyword matching")
        return _inferir_tema_fallback([], texto)


def _inferir_tema_fallback(keywords: list[str], texto: str) -> str:
    """
    Fallback theme inference using keyword matching.
    Used when BERTopic model is unavailable or returns low confidence.
    """
    TEMAS = {
        "economia": ["fiscal", "orçamento", "imposto", "tributo", "economia", "financeiro",
                     "dívida", "previdência", "benefício", "auxílio", "bolsa"],
        "saúde": ["saúde", "sus", "medicamento", "hospital", "doença", "médico",
                  "sanitário", "epidemia", "vacina", "enfermagem"],
        "educação": ["educação", "escola", "ensino", "universidade", "professor",
                     "aluno", "formação", "letramento", "mec"],
        "segurança pública": ["crime", "polícia", "penal", "prisão", "segurança",
                              "violência", "arma", "tráfico", "pena", "presídio"],
        "meio ambiente": ["ambiental", "floresta", "desmatamento", "carbono", "clima",
                          "biodiversidade", "resíduo", "saneamento", "água"],
        "direitos humanos": ["direito", "igualdade", "discriminação", "gênero",
                             "racial", "indígena", "criança", "idoso", "lgbtq"],
        "infraestrutura": ["infraestrutura", "rodovia", "ferrovia", "porto", "aeroporto",
                           "energia", "elétrica", "petróleo", "gás", "combustível"],
        "reforma política": ["eleição", "partido", "candidato", "voto", "urna",
                             "político", "mandato", "reforma", "constituição"],
        "agricultura": ["agropecuária", "rural", "produtor", "safra", "agrotóxico",
                        "irrigação", "crédito rural", "fundiário"],
    }

    texto_lower   = texto.lower()
    todas_palavras = keywords + re.findall(r'\b\w{5,}\b', texto_lower)

    pontuacao = {}
    for tema, termos in TEMAS.items():
        score = sum(1 for t in termos if any(t in p for p in todas_palavras))
        if score > 0:
            pontuacao[tema] = score

    if not pontuacao:
        return "outros"
    return max(pontuacao, key=pontuacao.get)


# =============================================================================
# 4. FULL NLP PIPELINE
# =============================================================================

def processar_votacao_nlp(
    ementa: str,
    descricao_votacao: str,
    resultado_aprovacao: Optional[bool] = None,
    tipo_proposicao: str = "",
    skip_summary: bool = False,
) -> dict:
    """
    Main NLP pipeline. Receives voting texts and returns summary, keywords and topic.

    Args:
        ementa:              Law summary text
        descricao_votacao:   Voting description text
        resultado_aprovacao: Whether the voting was approved
        tipo_proposicao:     Proposition type (PL, PEC, etc.)
        skip_summary:        If True, skips Ollama summary generation.
                             Set to True for uncategorized/procedural votings
                             to avoid wasting LLM tokens.

    Returns:
        {
            "resumo":          str,
            "keywords":        list[str],
            "tema_inferido":   str,
            "modelo_resumo":   str,
            "modelo_keywords": str,
        }
    """
    texto_completo = " ".join(filter(None, [ementa, descricao_votacao]))

    contexto = []
    if tipo_proposicao:
        contexto.append(f"Tipo da proposição: {tipo_proposicao}")
    if resultado_aprovacao is not None:
        status = "APROVADA" if resultado_aprovacao else "REJEITADA"
        contexto.append(f"Resultado da votação: {status}")

    logger.info("Iniciando pipeline NLP para votação...")

    keywords = extrair_keywords(texto_completo)

    # Skip summary for procedural/uncategorized votings — saves ~3 min per voting
    if skip_summary:
        resumo = ""
        logger.debug("Summary skipped (uncategorized voting)")
    else:
        resumo = gerar_resumo(texto_completo, contexto_extra="; ".join(contexto))

    tema_inferido = _inferir_tema_bertopic(texto_completo)

    return {
        "resumo":          resumo,
        "keywords":        keywords,
        "tema_inferido":   tema_inferido,
        "modelo_resumo":   OLLAMA_MODEL,
        "modelo_keywords": KEYWORD_MODEL,
    }


# =============================================================================
# UTILITIES
# =============================================================================

def _limpar_texto(texto: str) -> str:
    """Removes problematic characters and normalizes whitespace."""
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'[^\w\s\.,;:!?\-\(\)\/]', ' ', texto)
    return texto.strip()
