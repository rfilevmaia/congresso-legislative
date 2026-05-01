"""
core/stance_detector.py — Speech stance detection and vote-speech consistency analysis.

Responsibilities:
  1. Detect stance (favor/contra/neutro) in a speech regarding a topic
  2. Classify a law direction (expansao/restricao/neutro) regarding a topic
  3. Infer the deputy revealed position from vote + law direction
  4. Compute consistency score between speech stance and vote position
  5. Compute the Speech Fidelity Index (IFD) per deputy
"""

from __future__ import annotations
import re
import json
import time
from typing import Optional
from loguru import logger
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# =============================================================================
# HELPERS
# =============================================================================

def _ollama_generate(prompt: str, num_predict: int = 2000) -> str:
    """
    Sends a prompt to the local Ollama instance and returns the response text.
    Returns empty string on failure.
    """
    try:
        import ollama
        client   = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": num_predict},
        )
        return response.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""


def _extract_json(text: str) -> Optional[dict]:
    """
    Extracts the first JSON object found in a string.
    Handles cases where the model wraps JSON in markdown backticks.
    """
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def stance_para_int(stance: str) -> int:
    """Converts stance string to integer for arithmetic comparison."""
    return {"favor": 1, "neutro": 0, "contra": -1, "indefinido": 0}.get(
        (stance or "").lower().strip(), 0
    )


# =============================================================================
# 1. STANCE DETECTION IN SPEECHES
# =============================================================================

def detectar_stance(texto: str, tema: str) -> dict:
    """
    Detects whether a speech is FOR, AGAINST, or NEUTRAL regarding a topic.

    Uses Ollama (local LLM) to classify the stance — this is necessary because
    keyword matching alone cannot distinguish "I support education reform" from
    "I oppose this education reform bill".

    Args:
        texto: Speech transcription text
        tema:  Legislative topic (e.g. "educação", "saúde", "reforma tributária")

    Returns:
        {
            "stance":        "favor" | "contra" | "neutro" | "indefinido",
            "confianca":     float 0.0-1.0,
            "justificativa": str (one-sentence explanation)
        }
    """
    if not texto or not tema or len(texto.strip()) < 30:
        return {
            "stance":        "indefinido",
            "confianca":     0.0,
            "justificativa": "insufficient text",
        }

    prompt = f"""Você é um analista político especializado no Congresso Brasileiro.

Leia o trecho de discurso abaixo e determine se o parlamentar está:
- A FAVOR do tema "{tema}": defende, apoia, elogia, pede mais recursos ou atenção
- CONTRA o tema "{tema}": critica, se opõe, pede cortes, questiona necessidade
- NEUTRO: apenas descreve a situação sem tomar partido claro

Responda APENAS com um JSON válido, sem texto adicional:
{{"stance": "favor", "confianca": 0.85, "justificativa": "uma frase curta explicando"}}

DISCURSO:
{texto[:1500]}

JSON:"""

    raw    = _ollama_generate(prompt, num_predict=2000)
    result = _extract_json(raw)

    if result:
        stance = (result.get("stance") or "indefinido").lower().strip()
        if stance not in ("favor", "contra", "neutro"):
            stance = "indefinido"
        return {
            "stance":        stance,
            "confianca":     float(result.get("confianca", 0.5)),
            "justificativa": result.get("justificativa", ""),
        }

    return {
        "stance":        "indefinido",
        "confianca":     0.0,
        "justificativa": "model did not return valid JSON",
    }


# =============================================================================
# 2. LAW DIRECTION CLASSIFICATION
# =============================================================================

def classificar_direcao_lei(ementa: str, tema: str) -> str:
    """
    Determines whether a law EXPANDS or RESTRICTS rights/resources for a topic.

    This is the key step that gives meaning to a vote:
      - YES on an expansion law  → deputy is FOR the topic
      - YES on a restriction law → deputy is AGAINST the topic
      - NO  on an expansion law  → deputy is AGAINST the topic
      - NO  on a restriction law → deputy is FOR the topic

    Args:
        ementa: Law summary/description text
        tema:   Legislative topic being evaluated

    Returns:
        "expansao"  — law expands, increases or creates benefits for the topic
        "restricao" — law cuts, reduces or limits something related to the topic
        "neutro"    — law only regulates without clear expansion or restriction
    """
    if not ementa or not tema:
        return "neutro"

    prompt = f"""Analise esta ementa de lei e determine se ela é:
- "expansao": amplia, aumenta, fortalece ou cria benefícios relacionados a "{tema}"
- "restricao": corta, reduz, limita ou restringe algo relacionado a "{tema}"
- "neutro": apenas regula sem expandir nem restringir

Responda APENAS com uma palavra: expansao, restricao ou neutro

EMENTA: {ementa[:500]}
RESPOSTA:"""

    raw = _ollama_generate(prompt, num_predict=2000).lower()

    if "expan" in raw:
        return "expansao"
    if "restri" in raw:
        return "restricao"
    return "neutro"


# =============================================================================
# 3. REVEALED POSITION FROM VOTE
# =============================================================================

def posicao_revelada_pelo_voto(voto: str, direcao_lei: str) -> str:
    """
    Infers the deputy real political position from their vote + law direction.

    Logic matrix:
      SIM + expansao   → favor   (voted yes for expansion = supports topic)
      SIM + restricao  → contra  (voted yes for restriction = opposes topic)
      NÃO + expansao   → contra  (voted no for expansion = opposes topic)
      NÃO + restricao  → favor   (voted no for restriction = supports topic)
      abstencao/other  → neutro  (no clear position revealed)

    Args:
        voto:        Deputy vote string ("Sim", "Não", "Abstenção", etc.)
        direcao_lei: Law direction ("expansao", "restricao", "neutro")

    Returns:
        "favor" | "contra" | "neutro"
    """
    voto_lower = (voto or "").lower().strip()

    if voto_lower == "sim":
        if direcao_lei == "expansao":
            return "favor"
        if direcao_lei == "restricao":
            return "contra"
        return "neutro"

    if voto_lower in ("não", "nao"):
        if direcao_lei == "expansao":
            return "contra"
        if direcao_lei == "restricao":
            return "favor"
        return "neutro"

    # abstencao, obstrucao, art.17, etc. — no clear position
    return "neutro"


# =============================================================================
# 4. CONSISTENCY SCORE
# =============================================================================

def calcular_consistencia(
    stance_discurso:     str,
    posicao_voto:        str,
    confianca_stance:    float = 1.0,
    threshold_confianca: float = 0.60,
) -> dict:
    """
    Compares speech stance with revealed vote position.

    Args:
        stance_discurso:     Stance detected in speech ("favor"/"contra"/"neutro")
        posicao_voto:        Position revealed by vote ("favor"/"contra"/"neutro")
        confianca_stance:    Confidence of the stance detection (0.0-1.0)
        threshold_confianca: Minimum confidence to flag an alert

    Returns:
        {
            "consistente": bool | None  (None = not a meaningful comparison)
            "score":       float        (-1.0 = contradiction, 1.0 = alignment)
            "alerta":      bool         (True = high-confidence contradiction)
        }
    """
    s = stance_para_int(stance_discurso)
    v = stance_para_int(posicao_voto)

    # Skip if either side is neutral or undefined — no meaningful comparison
    if s == 0 or v == 0:
        return {"consistente": None, "score": 0.0, "alerta": False}

    score  = 1.0 if s == v else -1.0
    alerta = (
        score < 0 and
        confianca_stance >= threshold_confianca and
        stance_discurso  != "indefinido" and
        posicao_voto     != "neutro"
    )

    return {
        "consistente": score > 0,
        "score":       score,
        "alerta":      alerta,
    }


# =============================================================================
# 5. IFD — SPEECH FIDELITY INDEX PER DEPUTY
# =============================================================================

def calcular_ifd_deputado(
    deputado_id:      int,
    tema:             str = None,
    limite_pares:     int = 50,
) -> dict:
    """
    Computes the Speech Fidelity Index (IFD) for a deputy.

    IFD = consistent comparisons / total valid comparisons
    Range: 0.0 (always contradicts speeches) to 1.0 (always consistent)

    A comparison is valid only when:
      - The speech and the voting share the same topic (category_final = tema_inferido)
      - The speech stance is not "indefinido"
      - The revealed vote position is not "neutro"

    Args:
        deputado_id:  Deputy database ID
        tema:         Optional topic filter (e.g. "educação")
        limite_pares: Max speech-vote pairs to analyze per deputy.
                      Controls Ollama API cost — each pair requires 2 LLM calls.

    Returns:
        {
            "deputado_id":   int,
            "ifd":           float | None,
            "consistentes":  int,
            "total_validos": int,
            "alertas":       list[dict]
        }
    """
    from core.database import get_session
    from sqlalchemy import text

    session = get_session()
    try:
        topic_filter = "AND n.tema_inferido = :tema" if tema else ""
        params: dict = {"dep_id": deputado_id, "lim": limite_pares}
        if tema:
            params["tema"] = tema

        # Fetch speech-vote pairs that share the same topic
        rows = session.execute(text(f"""
            SELECT
                d.transcricao_limpa            AS discurso,
                d.category_final               AS tema_discurso,
                vt.voto,
                n.tema_inferido                AS tema_votacao,
                p.ementa,
                v.id                           AS votacao_id
            FROM discursos d
            JOIN votos vt       ON vt.deputado_id  = d.id_deputado
            JOIN votacoes v     ON v.id            = vt.votacao_id
            JOIN votacoes_nlp n ON n.votacao_id    = v.id
            LEFT JOIN proposicoes p ON p.id        = v.proposicao_id
           WHERE
                d.id_deputado     = :dep_id
                AND d.category_final IS NOT NULL
                AND d.category_final NOT IN ('outros', 'uncategorized')
                AND n.tema_inferido  NOT IN ('outros', 'uncategorized')
                AND d.category_final  = n.tema_inferido
                {topic_filter}
            LIMIT :lim
        """), params).mappings().all()

    finally:
        session.close()

    if not rows:
        return {
            "deputado_id":   deputado_id,
            "ifd":           None,
            "consistentes":  0,
            "total_validos": 0,
            "alertas":       [],
            "mensagem":      "No speech-vote pairs found for the same topic.",
        }

    consistentes  = 0
    total_validos = 0
    alertas       = []

    for r in rows:
        tema_atual = r["tema_discurso"]

        # Step 1: detect stance in the speech
        stance_result = detectar_stance(r["discurso"], tema_atual)
        if stance_result["stance"] == "indefinido":
            continue  # no clear position in this speech — skip

        # Step 2: classify law direction
        direcao = classificar_direcao_lei(r["ementa"] or "", tema_atual)

        # Step 3: infer revealed position from vote + law direction
        posicao_voto = posicao_revelada_pelo_voto(r["voto"], direcao)

        # Step 4: compare stance vs revealed position
        consistencia = calcular_consistencia(
            stance_discurso=stance_result["stance"],
            posicao_voto=posicao_voto,
            confianca_stance=stance_result["confianca"],
        )

        if consistencia["score"] == 0.0:
            continue  # not a meaningful comparison — skip

        total_validos += 1
        if consistencia["consistente"]:
            consistentes += 1

        # Collect high-confidence contradictions as alerts
        if consistencia["alerta"]:
            alertas.append({
                "votacao_id":      r["votacao_id"],
                "tema":            tema_atual,
                "stance_discurso": stance_result["stance"],
                "justificativa":   stance_result["justificativa"],
                "direcao_lei":     direcao,
                "voto":            r["voto"],
                "posicao_voto":    posicao_voto,
                "ementa":          (r["ementa"] or "")[:120],
            })

        time.sleep(0.2)  # brief pause between Ollama calls

    ifd = round(consistentes / total_validos, 4) if total_validos > 0 else None

    return {
        "deputado_id":   deputado_id,
        "ifd":           ifd,
        "consistentes":  consistentes,
        "total_validos": total_validos,
        "alertas":       alertas,
    }
