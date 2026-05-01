"""
agents/enrichment_agent.py — Enriquecimento profundo via texto integral.

Fluxo eficiente com status_enriquecimento:

  None / pendente  -> processar
  enriquecido      -> pular (ja tem texto integral)
  sem_texto        -> pular (ja tentou, nao existe texto)
  sem_votos        -> pular (votacao sem votos = sem valor analitico)

Antes de qualquer chamada HTTP, o agente:
  1. Verifica votos no banco (local) -> sem votos: marca e pula
  2. Verifica status salvo           -> enriquecido/sem_texto: pula
  3. So entao tenta as fontes de texto (HTTP)
"""

from __future__ import annotations
import re
import time
from typing import Optional
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.api_camara import _get, _session
from core.nlp_local import processar_votacao_nlp
from core.database import upsert_votacao_nlp, get_session as db_session
from models.schema import Votacao

STATUS_PENDENTE    = "pendente"
STATUS_ENRIQUECIDO = "enriquecido"
STATUS_SEM_TEXTO   = "sem_texto"
STATUS_SEM_VOTOS   = "sem_votos"


# -- Triagem local (sem HTTP) --------------------------------------------------

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
        logger.error(f"Erro ao marcar status {status} para {votacao_id}: {e}")
    finally:
        session.close()


# -- Extração de texto (HTTP) --------------------------------------------------

def _baixar_texto_url(url: str) -> Optional[str]:
    """Baixa HTML e extrai texto limpo. Ignora PDFs."""
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
        logger.debug(f"Falha ao baixar {url}: {e}")
        return None


def _obter_texto_completo(vot: Votacao) -> Optional[str]:
    prop = vot.proposicao
    id_prop = prop.id if prop else None

    # Fonte 1: url_inteiro_teor já no banco
    if prop and prop.url_inteiro_teor:
        texto = _baixar_texto_url(prop.url_inteiro_teor)
        if texto:
            return texto

    if not id_prop:
        return None

    # Fonte 2: busca dados completos da proposição via API
    # Isso recupera urlInteiroTeor que não foi salvo na coleta original
    try:
        data = _get(f"/proposicoes/{id_prop}")
        prop_dados = data.get("dados", {})

        # Save urlInteiroTeor to DB for future runs (avoid repeat API calls)
        url_teor = prop_dados.get("urlInteiroTeor")
        if url_teor:
            session = db_session()
            try:
                session.execute(
                    text("UPDATE proposicoes SET url_inteiro_teor = :url WHERE id = :id"),
                    {"url": url_teor, "id": id_prop}
                )
                session.commit()
            finally:
                session.close()

            texto = _baixar_texto_url(url_teor)
            if texto:
                return texto

        # Fonte 3: se for proposição procedimental (DTQ, REQ, EMA),
        # tenta a proposição principal referenciada
        sigla = prop_dados.get("siglaTipo", "")
        TIPOS_PROCEDIMENTAIS = {"DTQ", "REQ", "RPD", "EMA", "RCP", "REL"}

        if sigla in TIPOS_PROCEDIMENTAIS:
            uri_principal = prop_dados.get("uriPropPrincipal")
            if uri_principal:
                try:
                    id_principal = int(uri_principal.rstrip("/").split("/")[-1])
                    data_principal = _get(f"/proposicoes/{id_principal}")
                    url_principal = data_principal.get("dados", {}).get("urlInteiroTeor")
                    texto = _baixar_texto_url(url_principal)
                    if texto:
                        return texto
                except Exception as e:
                    logger.debug(f"Could not fetch principal proposition: {e}")

    except Exception as e:
        logger.debug(f"Error fetching proposition {id_prop} from API: {e}")

    return None

# -- Processamento de uma votacao ---------------------------------------------

def _enriquecer_uma(votacao_id: str, forcar: bool = False) -> str:
    """
    Retorna o status resultante:
      sem_votos / sem_texto / enriquecido / nao_encontrada
    """
    # Passo 1: tem votos? (local, rapido)
    if _contar_votos(votacao_id) == 0:
        _marcar_status(votacao_id, STATUS_SEM_VOTOS)
        logger.debug(f"{votacao_id} -> sem_votos")
        return STATUS_SEM_VOTOS

    # Passo 2: checar status salvo
    if not forcar:
        status = _get_status(votacao_id)
        if status in (STATUS_ENRIQUECIDO, STATUS_SEM_TEXTO, STATUS_SEM_VOTOS):
            logger.debug(f"{votacao_id} -> ja processado ({status}), pulando")
            return status

    # Passo 3: carregar dados e buscar texto
    # Initialize ALL variables before the try block
    # so they are always defined even if an exception occurs
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
            session.close()
            return "nao_encontrada"

        descricao = vot.descricao or ""
        aprovacao = vot.aprovacao

        if vot.proposicao:
            ementa    = vot.proposicao.ementa or ""
            tipo_prop = vot.proposicao.sigle_tipo or ""
            id_prop   = vot.proposicao.id
            url_banco = vot.proposicao.url_inteiro_teor or ""

    except Exception as e:
        logger.error(f"Error loading votacao {votacao_id} from DB: {e}")
    finally:
        session.close()

    # Fetch full text AFTER session is closed — uses id_prop and url_banco
    texto_integral = None

    if url_banco:
        texto_integral = _baixar_texto_url(url_banco)

    if not texto_integral and id_prop:
        try:
            data = _get(f"/proposicoes/{id_prop}")
            prop_dados = data.get("dados", {})
            url_teor = prop_dados.get("urlInteiroTeor")

            if url_teor:
                session2 = db_session()
                try:
                    session2.execute(
                        text("UPDATE proposicoes SET url_inteiro_teor = :url WHERE id = :id"),
                        {"url": url_teor, "id": id_prop}
                    )
                    session2.commit()
                finally:
                    session2.close()
                texto_integral = _baixar_texto_url(url_teor)

            if not texto_integral:
                sigla = prop_dados.get("siglaTipo", "")
                TIPOS_PROCEDIMENTAIS = {"DTQ", "REQ", "RPD", "EMA", "RCP", "REL"}
                if sigla in TIPOS_PROCEDIMENTAIS:
                    uri_principal = prop_dados.get("uriPropPrincipal")
                    if uri_principal:
                        id_principal = int(uri_principal.rstrip("/").split("/")[-1])
                        data_p = _get(f"/proposicoes/{id_principal}")
                        url_p  = data_p.get("dados", {}).get("urlInteiroTeor")
                        texto_integral = _baixar_texto_url(url_p)

        except Exception as e:
            logger.debug(f"Error fetching text for proposition {id_prop}: {e}")

    # Passo 4: enriquecer com campos extras da API
    try:
        prop_api         = _get(f"/proposicoes/{id_prop}").get("dados", {}) if id_prop else {}
        ementa_detalhada = prop_api.get("ementaDetalhada") or ""
        keywords_api     = prop_api.get("keywords") or ""
        temas_api        = ", ".join(t.get("nome", "") for t in (prop_api.get("temas") or []))
    except Exception:
        ementa_detalhada = keywords_api = temas_api = ""

    # Passo 5: montar texto
    texto_para_nlp = "\n\n".join(filter(None, [
        f"TEMAS: {temas_api}"                   if temas_api        else None,
        f"KEYWORDS: {keywords_api}"             if keywords_api     else None,
        f"EMENTA DETALHADA: {ementa_detalhada}" if ementa_detalhada else None,
        f"EMENTA: {ementa}"                     if ementa           else None,
        f"DESCRICAO: {descricao}"               if descricao        else None,
        f"TEXTO:\n{texto_integral[:3000]}"      if texto_integral   else None,
    ]))

    if not texto_para_nlp.strip():
        _marcar_status(votacao_id, STATUS_SEM_TEXTO)
        return STATUS_SEM_TEXTO
    
    

    resultado = processar_votacao_nlp(
        ementa=ementa,
        descricao_votacao=texto_para_nlp,
        resultado_aprovacao=aprovacao,
        tipo_proposicao=tipo_prop,
    )
    resultado["modelo_keywords"] += " [texto_integral]"

    upsert_votacao_nlp(votacao_id, resultado)
    _marcar_status(votacao_id, STATUS_ENRIQUECIDO)

    logger.info(
        f"OK {votacao_id} | tema={resultado['tema_inferido']} | "
        f"keywords={resultado['keywords'][:3]}"
    )
    return STATUS_ENRIQUECIDO


# -- Interface publica --------------------------------------------------------

class EnrichmentAgent:
    """
    Fluxo recomendado:

        agent = EnrichmentAgent()
        agent.triar_sem_votos()          # classifica rapidamente (so banco)
        agent.relatorio_status()         # panorama antes
        agent.enriquecer_pendentes(100)  # processa os que valem
        agent.relatorio_temas()          # resultado
    """

    def triar_sem_votos(self) -> int:
        """
        Marca como 'sem_votos' todas as votacoes sem votos no banco.
        100% local, sem HTTP. Roda em segundos independente do volume.
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
            logger.info(f"{result.rowcount} votacoes marcadas como sem_votos.")
            return result.rowcount
        finally:
            session.close()

    def enriquecer_pendentes(self, limite: int = 100) -> int:
        """
        Processa apenas votacoes COM votos que ainda nao foram enriquecidas
        ou estao classificadas como 'outros'.
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
                    OR n.tema_inferido = 'outros'
                GROUP BY v.id
                ORDER BY v.data DESC
                LIMIT :lim
            """), {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("Nenhuma votacao pendente.")
            return 0

        logger.info(f"Enriquecendo {len(ids)} votacoes...")
        return self._processar_lote(ids)

    def enriquecer_votacao(self, votacao_id: str) -> str:
        """Forca enriquecimento de uma votacao especifica."""
        return _enriquecer_uma(votacao_id, forcar=True)

    def relatorio_status(self):
        """Mostra distribuicao de status_enriquecimento."""
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
        """Distribuicao de temas — exclui votacoes sem votos."""
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

    def _processar_lote(self, ids: list, forcar: bool = False) -> int:
        c = {STATUS_ENRIQUECIDO: 0, STATUS_SEM_TEXTO: 0, STATUS_SEM_VOTOS: 0}
        for vid in tqdm(ids, desc="Enriquecendo"):
            try:
                status = _enriquecer_uma(vid, forcar=forcar)
                if status in c:
                    c[status] += 1
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Erro ao enriquecer {vid}: {e}")
        logger.info(
            f"Lote concluido -> enriquecidos={c[STATUS_ENRIQUECIDO]} | "
            f"sem_texto={c[STATUS_SEM_TEXTO]} | sem_votos={c[STATUS_SEM_VOTOS]}"
        )
        return c[STATUS_ENRIQUECIDO]


    def reprocessar_todas(self, limite: int = 9999) -> int:
        """Force reprocessing of all votacoes with votes, regardless of current status."""
        print("Right version")
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
