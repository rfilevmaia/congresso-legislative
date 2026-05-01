"""
agents/enrichment_agent.py — Deep enrichment via full text and cascading topic classification.

Status flow:
  None / pendente  -> process
  enriquecido      -> skip (already has full text)
  sem_texto        -> skip (already tried, no text available)
  sem_votos        -> skip (no votes = no analytical value)

Topic classification cascade (in priority order):
  Layer 1: Câmara API temas field       — most reliable, no LLM needed
  Layer 2: Câmara API keywords field    — semi-structured, BERTopic
  Layer 3: ementaDetalhada              — richer ementa, BERTopic
  Layer 4: ementa curta                 — short but usually available, BERTopic
  Layer 5: Ollama reads law text        — reads actual content and infers topic

Before any HTTP call, the agent:
  1. Checks votes in the database (local) -> no votes: mark and skip
  2. Checks saved status                  -> enriquecido/sem_texto: skip
  3. Only then attempts text sources (HTTP)
"""

from __future__ import annotations
import re
import time
from typing import Optional
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.api_camara import _get, _session
from core.nlp_local import processar_votacao_nlp, _inferir_tema_bertopic
from core.database import upsert_votacao_nlp, get_session as db_session
from models.schema import Votacao

STATUS_PENDENTE    = "pendente"
STATUS_ENRIQUECIDO = "enriquecido"
STATUS_SEM_TEXTO   = "sem_texto"
STATUS_SEM_VOTOS   = "sem_votos"


# =============================================================================
# LAYER 1 — CÂMARA API TEMA MAPPING
# Maps the official Câmara thematic classification to your category system.
# This is the highest-confidence source — no LLM needed.
# =============================================================================

CAMARA_TO_CATEGORY = {
    "educação":                                    "educação",
    "cultura":                                     "cultura",
    "saúde":                                       "saúde",
    "economia":                                    "economia e finanças",
    "finanças e tributação":                       "reforma tributária",
    "orçamento":                                   "economia e finanças",
    "meio ambiente":                               "meio ambiente",
    "agricultura, pecuária, pesca e extrativismo": "agricultura",
    "segurança pública":                           "segurança pública",
    "direitos humanos e minorias":                 "direitos sociais",
    "trabalho e emprego":                          "direitos sociais",
    "previdência e assistência social":            "previdência social",
    "comunicações":                                "cultura",
    "ciência, tecnologia e inovação":              "cultura",
    "relações exteriores e defesa nacional":       "política externa",
    "transportes e trânsito":                      "transporte e infraestrutura",
    "habitação":                                   "habitação",
    "energia, recursos hídricos e minerais":       "transporte e infraestrutura",
    "indústria, comércio e serviços":              "economia e finanças",
    "direito civil e processual civil":            "direitos sociais",
    "direito penal e processual penal":            "segurança pública",
    "direito e defesa do consumidor":              "direitos sociais",
    "direito constitucional":                      "direitos civis e liberdade de expressão",
    "direito eleitoral e partidos políticos":      "direitos civis e liberdade de expressão",
    "desporto e lazer":                            "cultura",
    "turismo":                                     "cultura",
    "família, criança, adolescente e idoso":       "direitos sociais",
    "mulher":                                      "direitos da mulher",
    "pessoas com deficiência":                     "pessoas com deficiência",
    "povos indígenas":                             "direitos sociais",
    "religião":                                    "religião",
    "política urbana":                             "habitação",
    "saneamento":                                  "meio ambiente",
    "radiodifusão":                                "cultura",
}

CANDIDATE_LABELS = [
    "educação",
    "saúde",
    "economia e finanças",
    "segurança pública",
    "meio ambiente",
    "agricultura",
    "transporte e infraestrutura",
    "direitos sociais",
    "política externa",
    "reforma tributária",
    "previdência social",
    "habitação",
    "cultura",
    "direitos civis e liberdade de expressão",
    "religião",
    "direitos da mulher",
    "pessoas com deficiência",
    "aborto e direito à vida",
    "pandemia e saúde pública",
    "crise climática",
]


def _classificar_por_temas_camara(temas_api: list) -> Optional[str]:
    for tema in temas_api:
        # API returns "tema" field, not "nome"
        nome = (tema.get("tema") or tema.get("nome") or "").lower().strip()
        category = CAMARA_TO_CATEGORY.get(nome)
        if category:
            logger.debug(f"Layer 1 match: '{nome}' → '{category}'")
            return category
    return None

# =============================================================================
# LAYER 5 — OLLAMA READS LAW TEXT AND INFERS TOPIC
# Reads the actual content of the law and reasons about the topic.
# More reliable than zero-shot label matching on empty responses.
# =============================================================================

def _classificar_por_resumo_ollama(texto: str) -> Optional[str]:
    """
    Layer 5: Asks Ollama to read the law text and directly infer the topic.

    More reliable than zero-shot label matching because the model reasons
    from actual content rather than trying to match a predefined list.
    Also instructs the model to return uncategorized for procedural votings
    (requerimentos, destaques, votações de pauta) that have no policy content.

    Args:
        texto: Full law text, ementaDetalhada, ementa or descricao (in priority order)

    Returns:
        Category string matching one of CANDIDATE_LABELS, or None if uncategorized.
    """
    if not texto or len(texto.strip()) < 20:
        return None

    labels_str = ", ".join(CANDIDATE_LABELS)

    prompt = f"""Você é um classificador legislativo brasileiro.

Leia o texto abaixo e responda com UMA ÚNICA categoria que melhor descreve o tema principal.
Escolha apenas entre estas opções: {labels_str}

Se o texto for apenas procedimental (requerimento, destaque, votação de pauta, emenda de redação),
responda exatamente: uncategorized

Responda APENAS com o nome exato da categoria, sem pontuação ou explicação adicional.

TEXTO:
{texto[:1500]}

CATEGORIA: /no_think"""

    try:
        from config import OLLAMA_BASE_URL, OLLAMA_MODEL
        import ollama

        client   = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 2000},  # high enough for qwen3 thinking + response
        )
        categoria = response.get("response", "").strip().lower()

        # Guard against empty response
        if not categoria:
            logger.debug("Layer 5 (Ollama) returned empty response")
            return None

        # Explicit procedural detection
        if "uncategorized" in categoria or "procedimental" in categoria:
            logger.debug(f"Layer 5 (Ollama) identified as procedural: '{categoria}'")
            return None

        # Match against candidate labels
        for label in CANDIDATE_LABELS:
            if label in categoria or categoria in label:
                logger.debug(f"Layer 5 (Ollama) match: '{categoria}' → '{label}'")
                return label

        logger.debug(f"Layer 5 (Ollama) no match for response: '{categoria}'")

    except Exception as e:
        logger.error(f"Ollama classification error: {e}")

    return None


# =============================================================================
# CASCADING CLASSIFIER
# =============================================================================

def classificar_votacao_cascata(
    temas_api:        list,
    keywords_api:     str,
    ementa_detalhada: str,
    ementa:           str,
    descricao:        str,
    texto_integral:   str = None,
) -> tuple[str, str]:
    """
    Classifies a voting topic using multiple sources in priority order.
    Stops at the first successful classification.

    Args:
        temas_api:        List of tema dicts from Câmara API
        keywords_api:     Keywords string from Câmara API
        ementa_detalhada: Detailed law summary from API
        ementa:           Short law summary
        descricao:        Voting description
        texto_integral:   Full law text downloaded from urlInteiroTeor

    Returns:
        (category, source) where source indicates which layer succeeded.
    """
    # Layer 1: Câmara official themes — highest confidence, no LLM
    if temas_api:
        cat = _classificar_por_temas_camara(temas_api)
        if cat:
            return cat, "camara_api"

    # Layer 2: Câmara API keywords → BERTopic
    if keywords_api and len(keywords_api) > 10:
        cat = _inferir_tema_bertopic(keywords_api)
        if cat and cat != "uncategorized":
            return cat, "keywords_api_bertopic"

    # Layer 3: ementaDetalhada → BERTopic
    if ementa_detalhada and len(ementa_detalhada) > 20:
        cat = _inferir_tema_bertopic(ementa_detalhada)
        if cat and cat != "uncategorized":
            return cat, "ementa_detalhada_bertopic"

    # Layer 4: short ementa → BERTopic
    if ementa and len(ementa) > 10:
        cat = _inferir_tema_bertopic(ementa)
        if cat and cat != "uncategorized":
            return cat, "ementa_bertopic"

    # Layer 5: Ollama reads actual law content and infers topic
    # Uses full text when available, falls back to ementa/descricao
    texto_para_ollama = texto_integral or " ".join(filter(None, [
        ementa_detalhada, ementa, descricao
    ]))
    if texto_para_ollama:
        cat = _classificar_por_resumo_ollama(texto_para_ollama)
        if cat:
            return cat, "ollama_resumo"

    return "uncategorized", "none"


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def _contar_votos(votacao_id: str) -> int:
    session = db_session()
    try:
        return session.execute(
            text("SELECT COUNT(*) FROM votos WHERE votacao_id = :vid"),
            {"vid": votacao_id}
        ).scalar()
    finally:
        session.close()


def _get_status(votacao_id: str) -> Optional[str]:
    session = db_session()
    try:
        row = session.execute(
            text("SELECT status_enriquecimento FROM votacoes_nlp WHERE votacao_id = :vid"),
            {"vid": votacao_id}
        ).fetchone()
        return row[0] if row else None
    finally:
        session.close()


def _marcar_status(votacao_id: str, status: str):
    session = db_session()
    try:
        existe = session.execute(
            text("SELECT id FROM votacoes_nlp WHERE votacao_id = :vid"),
            {"vid": votacao_id}
        ).fetchone()
        if existe:
            session.execute(
                text("UPDATE votacoes_nlp SET status_enriquecimento = :s WHERE votacao_id = :vid"),
                {"s": status, "vid": votacao_id}
            )
        else:
            session.execute(
                text("""
                    INSERT INTO votacoes_nlp (votacao_id, status_enriquecimento, processado_em)
                    VALUES (:vid, :s, NOW())
                """),
                {"vid": votacao_id, "s": status}
            )
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error marking status {status} for {votacao_id}: {e}")
    finally:
        session.close()


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def _baixar_texto_url(url: str) -> Optional[str]:
    """Downloads HTML and extracts clean text. Ignores PDFs."""
    if not url:
        return None
    if ".pdf" in url.lower() or "pdf" in url.lower():
        return None
    try:
        resp = _session.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        if "pdf" in resp.headers.get("Content-Type", "").lower():
            return None
        texto = resp.text
        if "<html" in texto.lower():
            texto = re.sub(r"<[^>]+>", " ", texto)
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)
        texto = texto.strip()
        return texto if len(texto) > 100 else None
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return None


# =============================================================================
# MAIN ENRICHMENT FUNCTION
# =============================================================================

def _enriquecer_uma(votacao_id: str, forcar: bool = False) -> str:
    """
    Enriches a single voting record with NLP and cascading topic classification.

    Returns the resulting status:
      sem_votos / sem_texto / enriquecido / nao_encontrada
    """
    # Step 1: check votes (local, fast)
    if _contar_votos(votacao_id) == 0:
        _marcar_status(votacao_id, STATUS_SEM_VOTOS)
        logger.debug(f"{votacao_id} -> sem_votos")
        return STATUS_SEM_VOTOS

    # Step 2: check saved status
    if not forcar:
        status = _get_status(votacao_id)
        if status in (STATUS_ENRIQUECIDO, STATUS_SEM_TEXTO, STATUS_SEM_VOTOS):
            logger.debug(f"{votacao_id} -> already processed ({status}), skipping")
            return status

    # Step 3: load data from DB — extract ALL values while session is open
    ementa    = ""
    descricao = ""
    tipo_prop = ""
    aprovacao = None
    id_prop   = None
    url_banco = ""

    session = db_session()
    try:
        vot = session.get(Votacao, votacao_id)
        if not vot:
            return "nao_encontrada"

        descricao = vot.descricao or ""
        aprovacao = vot.aprovacao

        if vot.proposicao:
            ementa    = vot.proposicao.ementa or ""
            tipo_prop = vot.proposicao.sigle_tipo or ""
            id_prop   = vot.proposicao.id
            url_banco = vot.proposicao.url_inteiro_teor or ""

    except Exception as e:
        logger.error(f"Error loading votacao {votacao_id}: {e}")
    finally:
        session.close()

    # Step 4: fetch full text and extra fields from API (after session is closed)
    texto_integral   = None
    ementa_detalhada = ""
    keywords_api     = ""
    temas_api        = []

    # Try url_inteiro_teor from DB first
    if url_banco:
        texto_integral = _baixar_texto_url(url_banco)

    # Fetch full proposition data from API
    if id_prop:
        try:
            data       = _get(f"/proposicoes/{id_prop}")
            prop_dados = data.get("dados", {})

            ementa_detalhada = prop_dados.get("ementaDetalhada") or ""
            keywords_api     = prop_dados.get("keywords") or ""
            temas_api        = prop_dados.get("temas") or []

            # Save urlInteiroTeor to DB for future runs
            url_teor = prop_dados.get("urlInteiroTeor")
            if url_teor and not url_banco:
                session2 = db_session()
                try:
                    session2.execute(
                        text("UPDATE proposicoes SET url_inteiro_teor = :url WHERE id = :id"),
                        {"url": url_teor, "id": id_prop}
                    )
                    session2.commit()
                finally:
                    session2.close()

                if not texto_integral:
                    texto_integral = _baixar_texto_url(url_teor)

            # For procedural propositions, try the principal proposition
            sigla = prop_dados.get("siglaTipo", "")
            TIPOS_PROCEDIMENTAIS = {"DTQ", "REQ", "RPD", "EMA", "RCP", "REL"}
            if sigla in TIPOS_PROCEDIMENTAIS:
                uri_principal = prop_dados.get("uriPropPrincipal")
                if uri_principal:
                    try:
                        id_principal = int(uri_principal.rstrip("/").split("/")[-1])
                        data_p       = _get(f"/proposicoes/{id_principal}")
                        prop_p       = data_p.get("dados", {})

                        # Inherit themes and keywords from principal proposition
                        if not temas_api:
                            temas_api = prop_p.get("temas") or []
                        if not ementa_detalhada:
                            ementa_detalhada = prop_p.get("ementaDetalhada") or ""
                        if not keywords_api:
                            keywords_api = prop_p.get("keywords") or ""

                        # Try to get full text from principal proposition
                        if not texto_integral:
                            url_p          = prop_p.get("urlInteiroTeor")
                            texto_integral = _baixar_texto_url(url_p)

                    except Exception as e:
                        logger.debug(f"Could not fetch principal proposition: {e}")

        except Exception as e:
            logger.debug(f"Error fetching proposition {id_prop} from API: {e}")

    # Step 5: cascade topic classification
    # texto_integral is passed so Layer 5 can read actual law content
    tema_inferido, fonte_tema = classificar_votacao_cascata(
        temas_api=temas_api,
        keywords_api=keywords_api,
        ementa_detalhada=ementa_detalhada,
        ementa=ementa,
        descricao=descricao,
        texto_integral=texto_integral,
    )
    logger.debug(f"{votacao_id} → tema='{tema_inferido}' via {fonte_tema}")

    # Step 6: build NLP text — use all available sources
    texto_para_nlp = "\n\n".join(filter(None, [
        f"TEMAS: {', '.join(t.get('nome', '') for t in temas_api)}" if temas_api        else None,
        f"KEYWORDS: {keywords_api}"                                  if keywords_api     else None,
        f"EMENTA DETALHADA: {ementa_detalhada}"                     if ementa_detalhada else None,
        f"EMENTA: {ementa}"                                          if ementa           else None,
        f"DESCRICAO: {descricao}"                                    if descricao        else None,
        f"TEXTO:\n{texto_integral[:3000]}"                           if texto_integral   else None,
    ]))

    if not texto_para_nlp.strip():
        _marcar_status(votacao_id, STATUS_SEM_TEXTO)
        logger.debug(f"{votacao_id} -> sem_texto")
        return STATUS_SEM_TEXTO

    # Step 7: run NLP pipeline — skip summary for uncategorized votings
    # This avoids wasting ~3 min of Ollama time on procedural votings with no topic
    resultado = processar_votacao_nlp(
        ementa=ementa,
        descricao_votacao=texto_para_nlp,
        resultado_aprovacao=aprovacao,
        tipo_proposicao=tipo_prop,
        skip_summary=(tema_inferido == "uncategorized"),
    )

    # Override tema_inferido with cascade result — more reliable than BERTopic alone
    resultado["tema_inferido"]   = tema_inferido
    resultado["modelo_keywords"] += f" [{fonte_tema}]"

    upsert_votacao_nlp(votacao_id, resultado)
    _marcar_status(votacao_id, STATUS_ENRIQUECIDO)

    logger.info(
        f"OK {votacao_id} | tema={tema_inferido} ({fonte_tema}) | "
        f"keywords={resultado['keywords'][:3]}"
    )
    return STATUS_ENRIQUECIDO


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

class EnrichmentAgent:
    """
    Recommended workflow:

        agent = EnrichmentAgent()
        agent.triar_sem_votos()
        agent.relatorio_status()
        agent.enriquecer_pendentes(100)
        agent.relatorio_temas()
        agent.relatorio_fontes_tema()
    """

    def triar_sem_votos(self) -> int:
        """
        Marks as sem_votos all votacoes without votes in the database.
        100% local, no HTTP. Runs in seconds regardless of volume.
        """
        session = db_session()
        try:
            result = session.execute(text("""
                UPDATE votacoes_nlp n
                SET status_enriquecimento = 'sem_votos'
                WHERE
                    n.status_enriquecimento IS NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM votos v WHERE v.votacao_id = n.votacao_id
                    )
            """))
            session.commit()
            logger.info(f"{result.rowcount} votacoes marked as sem_votos.")
            return result.rowcount
        finally:
            session.close()

    def enriquecer_pendentes(self, limite: int = 100) -> int:
        """
        Processes only votacoes WITH votes that have not been enriched yet
        or are classified as outros / uncategorized.
        """
        session = db_session()
        try:
            ids = [row[0] for row in session.execute(text("""
                SELECT v.id
                FROM votacoes v
                JOIN votos vt ON vt.votacao_id = v.id
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                WHERE
                    n.status_enriquecimento IS NULL
                    OR n.status_enriquecimento = 'pendente'
                    OR n.tema_inferido IN ('outros', 'uncategorized')
                    OR n.tema_inferido IS NULL
                GROUP BY v.id
                ORDER BY v.data DESC
                LIMIT :lim
            """), {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("No pending votacoes.")
            return 0

        logger.info(f"Enriching {len(ids)} votacoes...")
        return self._processar_lote(ids)

    def enriquecer_votacao(self, votacao_id: str) -> str:
        """Forces enrichment of a specific votacao."""
        return _enriquecer_uma(votacao_id, forcar=True)

    def reprocessar_todas(self, limite: int = 9999) -> int:
        """Forces reprocessing of all votacoes with votes."""
        session = db_session()
        try:
            ids = [row[0] for row in session.execute(text("""
                SELECT DISTINCT v.id
                FROM votacoes v
                JOIN votos vt ON vt.votacao_id = v.id
                ORDER BY v.id
                LIMIT :lim
            """), {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("No votacoes found.")
            return 0

        logger.info(f"Reprocessing {len(ids)} votacoes...")
        return self._processar_lote(ids, forcar=True)

    def relatorio_status(self):
        """Shows distribution of status_enriquecimento."""
        session = db_session()
        try:
            rows = session.execute(text("""
                SELECT
                    COALESCE(n.status_enriquecimento, '(sem NLP)') AS status,
                    COUNT(*) AS total
                FROM votacoes v
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                GROUP BY status ORDER BY total DESC
            """)).fetchall()
        finally:
            session.close()
        print(f"\n{'STATUS':<25} {'TOTAL':>7}")
        print("-" * 34)
        for s, t in rows:
            print(f"{s:<25} {t:>7}")
        print()

    def relatorio_temas(self):
        """Shows topic distribution — excludes votacoes without votes."""
        session = db_session()
        try:
            rows = session.execute(text("""
                SELECT
                    COALESCE(n.tema_inferido, '(sem NLP)') AS tema,
                    COUNT(*) AS total,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM votacoes v
                JOIN votos vt ON vt.votacao_id = v.id
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                GROUP BY tema ORDER BY total DESC
            """)).fetchall()
        finally:
            session.close()
        print(f"\n{'TEMA':<25} {'TOTAL':>7} {'%':>6}")
        print("-" * 40)
        for tema, total, pct in rows:
            print(f"{tema:<25} {total:>7} {pct:>5}%")
        print()

    def relatorio_fontes_tema(self):
        """
        Shows which classification layer was responsible for each topic assignment.
        Useful for auditing classification quality.
        """
        session = db_session()
        try:
            rows = session.execute(text("""
                SELECT
                    CASE
                        WHEN modelo_keywords LIKE '%camara_api%'                THEN 'Layer 1 — Câmara API'
                        WHEN modelo_keywords LIKE '%keywords_api_bertopic%'     THEN 'Layer 2 — Keywords BERTopic'
                        WHEN modelo_keywords LIKE '%ementa_detalhada_bertopic%' THEN 'Layer 3 — Ementa Detalhada'
                        WHEN modelo_keywords LIKE '%ementa_bertopic%'           THEN 'Layer 4 — Ementa BERTopic'
                        WHEN modelo_keywords LIKE '%ollama_resumo%'             THEN 'Layer 5 — Ollama Resumo'
                        ELSE 'Unknown'
                    END AS fonte,
                    COUNT(*) AS total,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM votacoes_nlp
                WHERE tema_inferido IS NOT NULL
                  AND tema_inferido NOT IN ('outros', 'uncategorized')
                GROUP BY fonte
                ORDER BY total DESC
            """)).fetchall()
        finally:
            session.close()

        print(f"\n{'SOURCE':<35} {'TOTAL':>7} {'%':>6}")
        print("-" * 50)
        for fonte, total, pct in rows:
            print(f"{fonte:<35} {total:>7} {pct:>5}%")
        print()

    def _processar_lote(self, ids: list, forcar: bool = False) -> int:
        c = {STATUS_ENRIQUECIDO: 0, STATUS_SEM_TEXTO: 0, STATUS_SEM_VOTOS: 0}
        for vid in tqdm(ids, desc="Enriching"):
            try:
                status = _enriquecer_uma(vid, forcar=forcar)
                if status in c:
                    c[status] += 1
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Error enriching {vid}: {e}")
        logger.info(
            f"Batch done -> enriched={c[STATUS_ENRIQUECIDO]} | "
            f"sem_texto={c[STATUS_SEM_TEXTO]} | sem_votos={c[STATUS_SEM_VOTOS]}"
        )
        return c[STATUS_ENRIQUECIDO]


if __name__ == "__main__":
    import sys
    agent = EnrichmentAgent()
    if len(sys.argv) == 2 and "-" in sys.argv[1]:
        print(agent.enriquecer_votacao(sys.argv[1]))
    else:
        limite = int(sys.argv[1]) if len(sys.argv) == 2 else 100
        agent.relatorio_status()
        agent.triar_sem_votos()
        agent.relatorio_status()
        agent.enriquecer_pendentes(limite)
        agent.relatorio_temas()
        agent.relatorio_fontes_tema()
