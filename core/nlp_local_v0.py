"""
core/nlp_local.py — NLP 100% local, otimizado para Apple Silicon (MPS).

Duas responsabilidades:
  1. Extração de palavras-chave  → BERTimbau Large via KeyBERT (MPS)
  2. Resumo em português         → Qwen2.5-7B (ou 14B) via Ollama (MLX backend)

Modelos recomendados por perfil de hardware:
  ┌─────────────────┬──────────────────────────────────┬──────────────┐
  │ Tarefa          │ Modelo                           │ RAM mínima   │
  ├─────────────────┼──────────────────────────────────┼──────────────┤
  │ Keywords        │ BERTimbau Large (neuralmind/)     │ 4 GB         │
  │ Keywords leve   │ DeBERTinha (Emanuel/debertinha)  │ 2 GB         │
  │ Resumo padrão   │ qwen2.5:7b  (via Ollama)         │ 8 GB         │
  │ Resumo robusto  │ qwen2.5:14b (via Ollama)         │ 16 GB        │
  └─────────────────┴──────────────────────────────────┴──────────────┘

Por que BERTimbau para keywords?
  - Treinado em 2.68B tokens do corpus BRWAC (maior corpus PT-BR aberto)
  - Estado-da-arte em NER, STS e RTE para português brasileiro
  - KeyBERT usa embeddings do encoder para calcular MMR (Max Marginal Relevance),
    selecionando keywords semanticamente diversas e representativas

Por que Qwen2.5 para resumo?
  - Excelente qualidade em português, supera Llama 3.1 8B no mesmo hardware
  - Benchmarks em Apple Silicon M2: ~150 tok/s com Ollama/MLX backend
  - Instrução-tuned, segue prompts em PT-BR sem fine-tuning adicional
"""

from __future__ import annotations
import re
from typing import Optional
from loguru import logger
from config import (
    KEYWORD_MODEL, KEYWORD_TOP_N, KEYWORD_DEVICE,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
)


# ── 1. EXTRAÇÃO DE KEYWORDS — BERTimbau / KeyBERT ────────────────────────────

_keybert_model = None   # lazy init — carrega na primeira chamada

# ── 4. INFERÊNCIA DE TEMA — BERTopic ─────────────────────────────────────────

import os
from bertopic import BERTopic

_bertopic_model = None
#model_path='/Volumes/LaCie/development/congresso/models/'

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

import pickle
import torch

def _get_bertopic() -> "BERTopic":
    global _bertopic_model
    if _bertopic_model is not None:
        return _bertopic_model

    import pickle

    model_path = "/Volumes/LaCie/development/congresso/models/bertopic_v3.pkl"
    logger.info(f"Loading BERTopic from:: {model_path}...")

    with open(model_path, "rb") as f:
        _bertopic_model = pickle.load(f)

    logger.info("✅ BERTopic loaded.")
    return _bertopic_model


def _inferir_tema_bertopic(texto: str) -> str:
    if not texto or len(texto.strip()) < 20:
        return "uncategorized"
    try:
        model      = _get_bertopic()
        topics, probs = model.transform([texto])
        topic_id   = topics[0]
        confidence = float(probs[0].max()) if hasattr(probs[0], "max") else float(probs[0])
        if confidence < 0.30:
            return _inferir_tema([], texto)   # fallback para keyword matching
        return TOPIC_TO_CATEGORY.get(topic_id, "uncategorized")
    except Exception as e:
        logger.error(f"BERTopic.transform error: {e} — falling back to keyword matching")
        return _inferir_tema([], texto)
        
def _get_keybert() -> "KeyBERT":
    """
    Inicializa KeyBERT com BERTimbau Large em MPS (Apple Silicon).
    Lazy para não bloquear o import quando o módulo não é usado.
    """
    global _keybert_model
    if _keybert_model is not None:
        return _keybert_model

    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    logger.info(f"Carregando modelo de keywords: {KEYWORD_MODEL} no device={KEYWORD_DEVICE}")

    # SentenceTransformer encapsula BERTimbau e usa o device MPS automaticamente
    st_model = SentenceTransformer(KEYWORD_MODEL, device=KEYWORD_DEVICE)
    _keybert_model = KeyBERT(model=st_model)

    logger.info("✅ Modelo de keywords carregado.")
    return _keybert_model


def extrair_keywords(texto: str, top_n: int = KEYWORD_TOP_N) -> list[str]:
    """
    Extrai as top-N palavras-chave de um texto em português usando BERTimbau.

    Algoritmo: MMR (Maximal Marginal Relevance) — equilibra relevância e diversidade.
    Suporta n-gramas de 1 a 3 tokens para capturar termos compostos.

    Args:
        texto:  Texto de entrada (ementa, descrição da votação, etc.)
        top_n:  Número de keywords a retornar (padrão: KEYWORD_TOP_N do config)

    Returns:
        Lista de strings com as keywords extraídas, ordenadas por relevância.
    """
    if not texto or len(texto.strip()) < 20:
        logger.warning("Texto muito curto para extração de keywords.")
        return []

    texto = _limpar_texto(texto)

    try:
        kw_model = _get_keybert()
        resultados = kw_model.extract_keywords(
            texto,
            keyphrase_ngram_range=(1, 3),   # uni, bi e trigramas
            stop_words=None,                 # KeyBERT não tem stopwords PT nativas; usar texto limpo
            use_mmr=True,                    # Max Marginal Relevance: diversidade semântica
            diversity=0.5,                   # 0 = redundante, 1 = muito diverso
            top_n=top_n,
        )
        keywords = [kw for kw, _score in resultados]
        logger.debug(f"Keywords extraídas: {keywords}")
        return keywords

    except Exception as e:
        logger.error(f"Erro na extração de keywords: {e}")
        return _fallback_keywords(texto, top_n)


def _fallback_keywords(texto: str, top_n: int) -> list[str]:
    """
    Fallback estatístico (TF simples) caso BERTimbau não esteja disponível.
    Usa apenas frequência de termos após remoção de stopwords básicas.
    """
    STOPWORDS_PT = {
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "e", "ou", "que", "se", "ao", "um", "uma", "para", "com", "por",
        "os", "as", "a", "o", "é", "são", "foi", "ser", "ter", "não",
    }
    words = re.findall(r'\b[a-záéíóúâêîôûãõàç]{4,}\b', texto.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS_PT:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return sorted_words[:top_n]


# ── 2. RESUMO — Qwen2.5 via Ollama (MLX backend no Apple Silicon) ─────────────

def gerar_resumo(
    texto: str,
    contexto_extra: str = "",
    modelo: str = OLLAMA_MODEL,
) -> str:
    """
    Gera um resumo em português de um texto legislativo usando Qwen2.5 via Ollama.

    O Ollama usa automaticamente o backend MLX no Apple Silicon, que é
    significativamente mais rápido que PyTorch MPS para inferência de LLMs
    (benchmarks: MLX ~230 tok/s vs MPS ~7-9 tok/s no M2 Ultra).

    Args:
        texto:          Texto a resumir (ementa + descrição da votação)
        contexto_extra: Informação adicional (ex: tipo da proposição, resultado)
        modelo:         Modelo Ollama a usar (padrão: qwen2.5:7b)

    Returns:
        String com o resumo gerado (1-3 parágrafos, em português).
    """
    if not texto or len(texto.strip()) < 30:
        return "Texto insuficiente para geração de resumo."

    texto = _limpar_texto(texto)
    prompt = _montar_prompt_resumo(texto, contexto_extra)

    try:
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(
            model=modelo,
            prompt=prompt,
            options={
                "temperature": 0.3,      # baixo para resumos factuais
                "top_p": 0.9,
                "num_predict": 2000,      # ~250 palavras máximo
            },
        )
        resumo = response.get("response", "").strip()
        logger.debug(f"Resumo gerado ({len(resumo)} chars) com {modelo}")
        return resumo

    except Exception as e:
        logger.error(f"Erro ao gerar resumo com Ollama ({modelo}): {e}")
        return _fallback_resumo(texto)


def _montar_prompt_resumo(texto: str, contexto_extra: str = "") -> str:
    """Monta o prompt instrucional em português para Qwen2.5."""
    ctx = f"\nContexto adicional: {contexto_extra}" if contexto_extra else ""
    return f"""Você é um analista legislativo especializado no Congresso Nacional Brasileiro.
Leia o texto abaixo e escreva um resumo claro e objetivo em português brasileiro.
O resumo deve ter no máximo 3 parágrafos, destacar o tema central, o objetivo da proposta
e eventuais impactos para a população. Use linguagem acessível, sem jargões desnecessários.{ctx}

TEXTO:
{texto[:3000]}

RESUMO:"""


def _fallback_resumo(texto: str) -> str:
    """Retorna as primeiras sentenças do texto como resumo de emergência."""
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    return " ".join(sentencas[:3])


# ── 3. PIPELINE NLP COMPLETO ──────────────────────────────────────────────────

def processar_votacao_nlp(
    ementa: str,
    descricao_votacao: str,
    resultado_aprovacao: Optional[bool] = None,
    tipo_proposicao: str = "",
) -> dict:
    """
    Função principal do pipeline NLP. Recebe os textos de uma votação e
    retorna um dict com resumo, keywords e tema inferido.

    Estrutura do retorno:
    {
        "resumo":          str,
        "keywords":        list[str],
        "tema_inferido":   str,
        "modelo_resumo":   str,
        "modelo_keywords": str,
    }
    """
    # Concatena os textos disponíveis para enriquecer o contexto
    texto_completo = " ".join(filter(None, [ementa, descricao_votacao]))

    contexto = []
    if tipo_proposicao:
        contexto.append(f"Tipo da proposição: {tipo_proposicao}")
    if resultado_aprovacao is not None:
        status = "APROVADA" if resultado_aprovacao else "REJEITADA"
        contexto.append(f"Resultado da votação: {status}")

    logger.info("Iniciando pipeline NLP para votação...")

    keywords = extrair_keywords(texto_completo)
    resumo = gerar_resumo(texto_completo, contexto_extra="; ".join(contexto))
    #tema_inferido = _inferir_tema(keywords, texto_completo)
    tema_inferido = _inferir_tema_bertopic(texto_completo)
    

    return {
        "resumo": resumo,
        "keywords": keywords,
        "tema_inferido": tema_inferido,
        "modelo_resumo": OLLAMA_MODEL,
        "modelo_keywords": KEYWORD_MODEL,
    }


def _inferir_tema(keywords: list[str], texto: str) -> str:
    """
    Infere o tema da votação a partir das keywords e mapeamento temático.
    Retorna o tema mais provável ou "outros" se não identificado.
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

    texto_lower = texto.lower()
    todas_palavras = keywords + re.findall(r'\b\w{5,}\b', texto_lower)

    pontuacao = {}
    for tema, termos in TEMAS.items():
        score = sum(1 for t in termos if any(t in p for p in todas_palavras))
        if score > 0:
            pontuacao[tema] = score

    if not pontuacao:
        return "outros"
    return max(pontuacao, key=pontuacao.get)


# ── Utilitários internos ──────────────────────────────────────────────────────

def _limpar_texto(texto: str) -> str:
    """Remove caracteres problemáticos e normaliza espaços."""
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'[^\w\s\.,;:!?\-\(\)\/]', ' ', texto)
    return texto.strip()
