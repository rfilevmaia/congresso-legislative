"""
agents/enrichment_agent.py — Agente de enriquecimento profundo via texto integral.

Problema que resolve:
  O NLPAgent processa apenas a ementa curta (1-2 linhas), o que resulta em
  keywords genéricas e tema "outros" para a maioria das votações.

Solução:
  1. Busca no banco as votações com tema "outros" ou sem keywords úteis
  2. Para cada votação, localiza a proposição associada
  3. Tenta obter o texto completo em 3 fontes (ordem de preferência):
       a) url_inteiro_teor já salva no banco (PDF ou HTML)
       b) API /proposicoes/{id}/textos  → lista de versões do texto
       c) API /votacoes/{id} → campo descricao + proposicaoObjeto
  4. Extrai o texto limpo (remove cabeçalhos jurídicos, artigos de forma)
  5. Roda NLP completo com o texto enriquecido via Qwen3.5 local
  6. Atualiza votacoes_nlp com resumo, keywords e tema mais precisos

Uso:
    from agents.enrichment_agent import EnrichmentAgent

    agent = EnrichmentAgent()
    agent.enriquecer_sem_tema(limite=100)          # foca nos "outros"
    agent.enriquecer_votacao("2437698-2")          # uma votação específica
    agent.enriquecer_todas(limite=200)             # reprocessa tudo
"""

from __future__ import annotations
import re
import time
from typing import Optional
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.api_camara import _get, _session, CAMARA_API_BASE
from core.nlp_local import processar_votacao_nlp
from core.database import upsert_votacao_nlp, get_session as db_session
from models.schema import Votacao, Proposicao, VotacaoNLP


# ── Extração de texto das fontes disponíveis ──────────────────────────────────

def _buscar_textos_api(id_proposicao: int) -> list[dict]:
    """
    Chama GET /proposicoes/{id}/textos.
    Retorna lista com campos: tipo, dataTexto, urlTexto, uriTexto
    """
    try:
        data = _get(f"/proposicoes/{id_proposicao}/textos")
        return data.get("dados", [])
    except Exception as e:
        logger.debug(f"Sem textos via API para proposição {id_proposicao}: {e}")
        return []


def _buscar_inteiro_teor_api(id_proposicao: int) -> Optional[str]:
    """
    Tenta obter o campo urlInteiroTeor diretamente dos detalhes da proposição.
    Útil quando o campo não foi salvo no banco na coleta original.
    """
    try:
        data = _get(f"/proposicoes/{id_proposicao}")
        prop = data.get("dados", {})
        return prop.get("urlInteiroTeor") or prop.get("url_inteiro_teor")
    except Exception as e:
        logger.debug(f"Erro ao buscar detalhes da proposição {id_proposicao}: {e}")
        return None


def _baixar_texto_url(url: str) -> Optional[str]:
    """
    Baixa o conteúdo de uma URL (HTML ou texto puro).
    PDFs são ignorados — extração de PDF requer dependência extra (pdfplumber).
    Retorna o texto limpo ou None se falhar / for PDF.
    """
    if not url:
        return None

    # PDFs precisam de tratamento especial — por ora, pulamos
    if url.lower().endswith(".pdf") or "pdf" in url.lower():
        logger.debug(f"URL é PDF, pulando extração direta: {url}")
        return None

    try:
        resp = _session.get(url, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" in content_type:
            logger.debug(f"Resposta é PDF, pulando: {url}")
            return None

        texto = resp.text
        # Se for HTML, remove tags
        if "<html" in texto.lower() or "<body" in texto.lower():
            texto = re.sub(r"<[^>]+>", " ", texto)

        return _limpar_texto_juridico(texto)

    except Exception as e:
        logger.debug(f"Falha ao baixar {url}: {e}")
        return None


def _limpar_texto_juridico(texto: str) -> str:
    """
    Remove ruídos comuns em textos legislativos:
    cabeçalhos repetitivos, números de artigo isolados,
    espaços múltiplos e linhas em branco excessivas.
    """
    # Remove sequências de espaços/tabs
    texto = re.sub(r"[ \t]+", " ", texto)
    # Remove linhas que só têm números (ex: numeração de artigos)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)
    # Remove cabeçalhos de diário oficial
    texto = re.sub(r"DIÁRIO OFICIAL.*?\n", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"Câmara dos Deputados.*?\n", "", texto, flags=re.IGNORECASE)
    # Compacta múltiplas linhas em branco
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _montar_texto_enriquecido(
    ementa: str,
    descricao_votacao: str,
    texto_integral: str,
) -> str:
    """
    Combina as fontes de texto em ordem de riqueza semântica.
    Limita o total a ~4000 chars para não estourar o contexto do Qwen3.5.
    """
    partes = []

    if ementa:
        partes.append(f"EMENTA: {ementa}")

    if descricao_votacao:
        partes.append(f"DESCRIÇÃO DA VOTAÇÃO: {descricao_votacao}")

    if texto_integral:
        # Pega os primeiros 3000 chars do texto integral — contém o essencial
        partes.append(f"TEXTO DA PROPOSIÇÃO:\n{texto_integral[:3000]}")

    return "\n\n".join(partes)


# ── Lógica principal de enriquecimento ───────────────────────────────────────

def _obter_texto_completo(votacao: Votacao) -> Optional[str]:
    """
    Tenta obter o texto completo da proposição associada a uma votação.
    Percorre as fontes em ordem de preferência e retorna o primeiro sucesso.
    """
    prop: Optional[Proposicao] = votacao.proposicao
    id_prop = prop.id if prop else None

    # Fonte 1: url_inteiro_teor já no banco
    if prop and prop.url_inteiro_teor:
        texto = _baixar_texto_url(prop.url_inteiro_teor)
        if texto and len(texto) > 100:
            logger.debug(f"Texto obtido via url_inteiro_teor do banco (prop {id_prop})")
            return texto

    # Fonte 2: API /proposicoes/{id}/textos
    if id_prop:
        textos_api = _buscar_textos_api(id_prop)
        for t in textos_api:
            url = t.get("urlTexto") or t.get("uriTexto")
            texto = _baixar_texto_url(url)
            if texto and len(texto) > 100:
                logger.debug(f"Texto obtido via /textos da API (prop {id_prop})")
                return texto
            time.sleep(0.5)

    # Fonte 3: url_inteiro_teor via API (caso não tenha sido salvo no banco)
    if id_prop:
        url_api = _buscar_inteiro_teor_api(id_prop)
        if url_api:
            texto = _baixar_texto_url(url_api)
            if texto and len(texto) > 100:
                logger.debug(f"Texto obtido via urlInteiroTeor da API (prop {id_prop})")
                return texto

    logger.debug(f"Nenhum texto integral encontrado para votação {votacao.id}")
    return None


def _enriquecer_uma(votacao_id: str, forcar: bool = False) -> bool:
    """
    Busca texto completo e roda NLP enriquecido para uma votação.
    Retorna True se processou com sucesso.
    """
    session = db_session()
    try:
        vot = session.get(Votacao, votacao_id)
        if not vot:
            logger.warning(f"Votação {votacao_id} não encontrada no banco.")
            return False

        # Verifica se já tem NLP enriquecido (tema != "outros") — pula se não forcar
        if not forcar and vot.nlp and vot.nlp.tema_inferido not in ("outros", None, ""):
            logger.debug(f"Votação {votacao_id} já tem tema definido. Pulando.")
            return False

        ementa = (vot.proposicao.ementa or "") if vot.proposicao else ""
        descricao = vot.descricao or ""
        tipo_prop = (vot.proposicao.sigle_tipo or "") if vot.proposicao else ""

    finally:
        session.close()

    # Busca texto integral (fora da session para não bloquear conexão durante HTTP)
    session2 = db_session()
    try:
        vot2 = session2.get(Votacao, votacao_id)
        texto_integral = _obter_texto_completo(vot2)
    finally:
        session2.close()

    # Monta texto combinado
    texto_completo = _montar_texto_enriquecido(ementa, descricao, texto_integral)

    if not texto_completo.strip():
        logger.warning(f"Votação {votacao_id} sem texto suficiente para NLP.")
        return False

    tem_texto_extra = texto_integral and len(texto_integral) > 100
    logger.info(
        f"Enriquecendo {votacao_id} | "
        f"texto_integral={'SIM' if tem_texto_extra else 'NÃO'} | "
        f"{len(texto_completo)} chars"
    )

    # Roda NLP com texto enriquecido
    resultado = processar_votacao_nlp(
        ementa=ementa,
        descricao_votacao=texto_completo,   # passa tudo como descrição enriquecida
        resultado_aprovacao=None,
        tipo_proposicao=tipo_prop,
    )

    # Marca que foi enriquecido com texto integral
    if tem_texto_extra:
        resultado["modelo_keywords"] = resultado["modelo_keywords"] + " [texto_integral]"

    upsert_votacao_nlp(votacao_id, resultado)
    logger.info(
        f"✅ {votacao_id} → tema={resultado['tema_inferido']} | "
        f"keywords={resultado['keywords'][:4]}"
    )
    return True


# ── Interface pública do agente ───────────────────────────────────────────────

class EnrichmentAgent:
    """
    Agente de enriquecimento profundo de votações.

    Foca nas votações classificadas como "outros" ou sem keywords relevantes,
    buscando o texto completo da proposição para melhorar o NLP.

    Uso:
        agent = EnrichmentAgent()

        # Processar votações sem tema definido (mais comum)
        agent.enriquecer_sem_tema(limite=100)

        # Reprocessar uma votação específica pelo ID
        agent.enriquecer_votacao("2437698-2")

        # Reprocessar todas, inclusive as já classificadas
        agent.enriquecer_todas(limite=200)

        # Ver distribuição de temas atual
        agent.relatorio_temas()
    """

    def enriquecer_sem_tema(self, limite: int = 100) -> int:
        """
        Processa votações com tema 'outros', sem NLP, ou com keywords vazias.
        É a operação mais comum — foca onde o enriquecimento mais ajuda.
        """
        session = db_session()
        try:
            sql = text("""
                SELECT v.id
                FROM votacoes v
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                WHERE
                    n.id IS NULL                          -- sem NLP ainda
                    OR n.tema_inferido = 'outros'         -- classificado como outros
                    OR n.tema_inferido IS NULL
                    OR n.keywords = '[]'::jsonb           -- sem keywords
                    OR jsonb_array_length(n.keywords) < 2 -- keywords insuficientes
                ORDER BY v.data DESC
                LIMIT :lim
            """)
            ids = [row[0] for row in session.execute(sql, {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("Nenhuma votação pendente de enriquecimento.")
            return 0

        logger.info(f"Enriquecendo {len(ids)} votações sem tema definido...")
        return self._processar_lote(ids)

    def enriquecer_votacao(self, votacao_id: str) -> bool:
        """Enriquece uma votação específica pelo ID (sempre reprocessa)."""
        return _enriquecer_uma(votacao_id, forcar=True)

    def enriquecer_todas(self, limite: int = 200) -> int:
        """Reprocessa todas as votações, inclusive as já classificadas."""
        session = db_session()
        try:
            sql = text("""
                SELECT id FROM votacoes
                ORDER BY data DESC
                LIMIT :lim
            """)
            ids = [row[0] for row in session.execute(sql, {"lim": limite})]
        finally:
            session.close()

        logger.info(f"Reprocessando {len(ids)} votações...")
        return self._processar_lote(ids, forcar=True)

    def relatorio_temas(self):
        """Imprime a distribuição atual de temas no banco."""
        session = db_session()
        try:
            sql = text("""
                SELECT
                    COALESCE(n.tema_inferido, '(sem NLP)') AS tema,
                    COUNT(*) AS total,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM votacoes v
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                GROUP BY tema
                ORDER BY total DESC
            """)
            rows = session.execute(sql).fetchall()
        finally:
            session.close()

        print(f"\n{'='*45}")
        print(f"{'TEMA':<25} {'TOTAL':>7} {'%':>6}")
        print(f"{'='*45}")
        for tema, total, pct in rows:
            print(f"{tema:<25} {total:>7} {pct:>5}%")
        print(f"{'='*45}\n")

    def _processar_lote(self, ids: list[str], forcar: bool = False) -> int:
        processadas = 0
        for vid in tqdm(ids, desc="Enriquecendo"):
            try:
                if _enriquecer_uma(vid, forcar=forcar):
                    processadas += 1
                time.sleep(0.3)   # cortesia com a API da Câmara
            except Exception as e:
                logger.error(f"Erro ao enriquecer votação {vid}: {e}")

        logger.info(f"✅ Enriquecimento concluído: {processadas}/{len(ids)} votações.")
        return processadas


# ── Execução direta ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    agent = EnrichmentAgent()

    if len(sys.argv) == 2 and not sys.argv[1].isdigit():
        # ID específico: python -m agents.enrichment_agent 2437698-2
        ok = agent.enriquecer_votacao(sys.argv[1])
        print("✅ Enriquecido." if ok else "⚠️ Sem alteração.")
    else:
        limite = int(sys.argv[1]) if len(sys.argv) == 2 else 100
        agent.relatorio_temas()
        agent.enriquecer_sem_tema(limite=limite)
        agent.relatorio_temas()
