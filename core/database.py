"""
core/database.py — Operações de persistência com PostgreSQL via SQLAlchemy.

Funções organizadas por entidade:
  upsert_deputado()            → insere ou atualiza deputado
  upsert_proposicao()          → insere ou atualiza proposição
  upsert_votacao()             → insere ou atualiza votação
  upsert_voto()                → insere ou atualiza voto individual
  upsert_votacao_nlp()         → salva análise NLP de uma votação
  query_deputado_votacoes()    → consulta cruzada principal (deputado × votação × voto × aprovação)
  query_votacoes_por_tema()    → busca votações por tema ou keyword
  query_deputados_por_tema()   → ranking de deputados por posição em um tema
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.schema import (
    get_session, Deputado, Proposicao, Votacao, Voto, VotacaoNLP
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_data(valor) -> Optional[datetime]:
    """Converte string ISO ou datetime para datetime, ou None."""
    if not valor:
        return None
    if isinstance(valor, datetime):
        return valor
    try:
        return datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    except Exception:
        return None


# ── UPSERTS ───────────────────────────────────────────────────────────────────

def upsert_deputado(dados: dict) -> int:
    """
    Insere ou atualiza um deputado. Retorna o ID.
    Aceita tanto o formato resumido (lista /deputados) quanto o completo (/deputados/{id}).
    """
    session = get_session()
    try:
        id_dep = int(dados.get("id", 0))
        if not id_dep:
            raise ValueError("Deputado sem ID válido.")

        dep = session.get(Deputado, id_dep)
        if dep is None:
            dep = Deputado(id=id_dep)
            session.add(dep)

        dep.nome_civil      = dados.get("nomeCivil") or dados.get("nome", "")
        dep.nome_eleitoral  = dados.get("nomeEleitoral") or dados.get("nome", "")
        dep.partido         = dados.get("siglaPartido") or (dados.get("ultimoStatus") or {}).get("siglaPartido", "")
        dep.uf              = dados.get("siglaUf") or (dados.get("ultimoStatus") or {}).get("siglaUf", "")
        dep.legislatura     = dados.get("idLegislatura") or (dados.get("ultimoStatus") or {}).get("idLegislatura")
        dep.uri             = dados.get("uri", "")
        dep.foto_url        = dados.get("urlFoto") or (dados.get("ultimoStatus") or {}).get("urlFoto", "")
        dep.atualizado_em   = datetime.utcnow()

        session.commit()
        return id_dep
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao upsert deputado {dados.get('id')}: {e}")
        raise
    finally:
        session.close()


def upsert_proposicao(dados: dict) -> Optional[int]:
    """
    Insere ou atualiza uma proposição. Retorna o ID ou None se dados insuficientes.
    """
    id_prop = dados.get("id")
    if not id_prop:
        return None

    session = get_session()
    try:
        prop = session.get(Proposicao, int(id_prop))
        if prop is None:
            prop = Proposicao(id=int(id_prop))
            session.add(prop)

        prop.uri            = dados.get("uri", "")
        prop.sigle_tipo     = dados.get("siglaTipo", "")
        prop.numero         = dados.get("numero")
        prop.ano            = dados.get("ano")
        prop.ementa         = dados.get("ementa") or dados.get("ementaDetalhada", "")
        prop.keywords_api   = dados.get("keywords") or []
        prop.temas_api      = [t.get("nome") for t in (dados.get("temas") or [])]
        prop.url_inteiro_teor = dados.get("urlInteiroTeor", "")

        session.commit()
        return int(id_prop)
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao upsert proposição {id_prop}: {e}")
        return None
    finally:
        session.close()


def upsert_votacao(dados: dict, proposicao_id: Optional[int] = None) -> str:
    """
    Insere ou atualiza uma votação. Retorna o ID (string).

    O campo 'aprovacao' da API pode ser booleano ou string ("Aprovado"/"Rejeitado").
    Esta função normaliza para bool.
    """
    id_vot = dados.get("id")
    if not id_vot:
        raise ValueError("Votação sem ID.")

    # Normaliza aprovação
    aprovacao_raw = dados.get("aprovacao")
    if isinstance(aprovacao_raw, bool):
        aprovacao = aprovacao_raw
    elif isinstance(aprovacao_raw, str):
        aprovacao = aprovacao_raw.lower() in ("sim", "aprovado", "true", "1")
    else:
        aprovacao = None

    # Extrai placar do campo 'placar' (dict) ou campos separados
    placar = dados.get("placar") or {}

    session = get_session()
    try:
        vot = session.get(Votacao, str(id_vot))
        if vot is None:
            vot = Votacao(id=str(id_vot))
            session.add(vot)

        vot.uri             = dados.get("uri", "")
        vot.data            = _parse_data(dados.get("data") or dados.get("dataHoraRegistro"))
        vot.descricao       = dados.get("descricao", "")
        vot.aprovacao       = aprovacao
        vot.placar_sim      = placar.get("votosSim") or dados.get("votosSim")
        vot.placar_nao      = placar.get("votosNao") or dados.get("votosNao")
        vot.placar_abstencao = placar.get("votosAbstencao") or dados.get("votosAbstencao")
        vot.id_orgao        = str(dados.get("idOrgao", "") or "")
        vot.id_evento       = str(dados.get("idEvento", "") or "")
        vot.proposicao_id   = proposicao_id

        session.commit()
        return str(id_vot)
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao upsert votação {id_vot}: {e}")
        raise
    finally:
        session.close()


def upsert_voto(votacao_id: str, deputado_id: int, tipo_voto: str, hora: str = None):
    """
    Insere ou atualiza o voto de um deputado em uma votação.
    Usa INSERT ... ON CONFLICT para performance em lotes grandes.
    """
    session = get_session()
    try:
        stmt = (
            pg_insert(Voto)
            .values(
                votacao_id=votacao_id,
                deputado_id=deputado_id,
                voto=tipo_voto,
                hora_registro=_parse_data(hora),
            )
            .on_conflict_do_update(
                constraint="uq_voto_votacao_deputado",
                set_={"voto": tipo_voto, "hora_registro": _parse_data(hora)},
            )
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao upsert voto dep={deputado_id} vot={votacao_id}: {e}")
    finally:
        session.close()


def upsert_votacao_nlp(votacao_id: str, nlp_data: dict):
    """
    Salva (ou atualiza) o resultado do pipeline NLP para uma votação.

    nlp_data deve conter as chaves retornadas por nlp_local.processar_votacao_nlp().
    """
    session = get_session()
    try:
        nlp = session.query(VotacaoNLP).filter_by(votacao_id=votacao_id).first()
        if nlp is None:
            nlp = VotacaoNLP(votacao_id=votacao_id)
            session.add(nlp)

        nlp.resumo          = nlp_data.get("resumo", "")
        nlp.keywords        = nlp_data.get("keywords", [])
        nlp.tema_inferido   = nlp_data.get("tema_inferido", "")
        nlp.sentimento_tema = nlp_data.get("sentimento_tema", "")
        nlp.modelo_resumo   = nlp_data.get("modelo_resumo", "")
        nlp.modelo_keywords = nlp_data.get("modelo_keywords", "")
        nlp.processado_em   = datetime.utcnow()

        session.commit()
        logger.info(f"NLP salvo para votação {votacao_id}")
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao salvar NLP da votação {votacao_id}: {e}")
    finally:
        session.close()


def salvar_votos_em_lote(votacao_id: str, votos_raw: list[dict]):
    """
    Persiste todos os votos de uma votação de forma eficiente.
    Garante que o deputado exista antes de inserir o voto.
    """
    from core.api_camara import obter_deputado
    session = get_session()
    inseridos = 0

    for v in votos_raw:
        dep_raw = v.get("deputado_") or v.get("deputado") or {}
        dep_id = dep_raw.get("id")
        if not dep_id:
            continue

        # Garante que o deputado exista no banco
        if not session.get(Deputado, int(dep_id)):
            session.close()
            upsert_deputado(dep_raw)
            session = get_session()

        tipo_voto = v.get("tipoVoto", "")
        hora = v.get("dataRegistroVoto")
        upsert_voto(votacao_id, int(dep_id), tipo_voto, hora)
        inseridos += 1

    logger.info(f"✅ {inseridos} votos salvos para votação {votacao_id}")
    session.close()


# ── CONSULTAS ─────────────────────────────────────────────────────────────────

def query_deputado_votacoes(
    nome_deputado: str = None,
    deputado_id: int = None,
    tema_keyword: str = None,
    data_inicio: str = None,
    data_fim: str = None,
) -> list[dict]:
    """
    Consulta principal: cruza deputado × votação × voto × aprovação × tema NLP.

    Retorna lista de dicts no formato:
    {
        "deputado_id":      int,
        "nome_deputado":    str,
        "partido":          str,
        "uf":               str,
        "votacao_id":       str,
        "data_votacao":     datetime,
        "descricao":        str,
        "voto":             str,          # "Sim" | "Não" | "Abstenção" | ...
        "aprovacao":        bool | None,
        "tema_inferido":    str,
        "keywords":         list[str],
        "resumo":           str,
    }
    """
    session = get_session()
    try:
        filtros = ["1=1"]
        params: dict = {}

        if nome_deputado:
            filtros.append(
                "(LOWER(d.nome_eleitoral) LIKE :nome OR LOWER(d.nome_civil) LIKE :nome)"
            )
            params["nome"] = f"%{nome_deputado.lower()}%"

        if deputado_id:
            filtros.append("d.id = :dep_id")
            params["dep_id"] = deputado_id

        if tema_keyword:
            filtros.append(
                "(LOWER(n.tema_inferido) LIKE :tema OR n.keywords::text LIKE :tema)"
            )
            params["tema"] = f"%{tema_keyword.lower()}%"

        if data_inicio:
            filtros.append("v.data >= :dt_inicio")
            params["dt_inicio"] = data_inicio

        if data_fim:
            filtros.append("v.data <= :dt_fim")
            params["dt_fim"] = data_fim

        where = " AND ".join(filtros)
        sql = text(f"""
            SELECT
                d.id            AS deputado_id,
                d.nome_eleitoral AS nome_deputado,
                d.partido,
                d.uf,
                v.id            AS votacao_id,
                v.data          AS data_votacao,
                v.descricao,
                vt.voto,
                v.aprovacao,
                n.tema_inferido,
                n.keywords,
                n.resumo
            FROM votos vt
            JOIN deputados d   ON d.id  = vt.deputado_id
            JOIN votacoes v    ON v.id  = vt.votacao_id
            LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
            WHERE {where}
            ORDER BY v.data DESC
        """)

        rows = session.execute(sql, params).mappings().all()
        return [dict(r) for r in rows]

    finally:
        session.close()


def query_votacoes_por_tema(tema: str, limit: int = 50) -> list[dict]:
    """
    Retorna votações cujo tema inferido ou keywords contenham o termo buscado.
    Ideal para visualizações e relatórios temáticos.
    """
    session = get_session()
    try:
        sql = text("""
            SELECT
                v.id, v.data, v.descricao, v.aprovacao,
                v.placar_sim, v.placar_nao,
                n.tema_inferido, n.keywords, n.resumo,
                p.sigle_tipo || ' ' || p.numero || '/' || p.ano AS proposicao
            FROM votacoes v
            LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
            LEFT JOIN proposicoes p  ON p.id = v.proposicao_id
            WHERE
                LOWER(n.tema_inferido) LIKE :tema
                OR n.keywords::text LIKE :tema
                OR LOWER(v.descricao) LIKE :tema
            ORDER BY v.data DESC
            LIMIT :lim
        """)
        rows = session.execute(sql, {"tema": f"%{tema.lower()}%", "lim": limit}).mappings().all()
        return [dict(r) for r in rows]
    finally:
        session.close()


def query_deputados_por_tema(tema: str) -> list[dict]:
    """
    Para um dado tema, mostra como cada partido/deputado votou.
    Útil para análise de alinhamento político.
    """
    session = get_session()
    try:
        sql = text("""
            SELECT
                d.nome_eleitoral,
                d.partido,
                d.uf,
                COUNT(*) FILTER (WHERE vt.voto = 'Sim')        AS votos_sim,
                COUNT(*) FILTER (WHERE vt.voto = 'Não')        AS votos_nao,
                COUNT(*) FILTER (WHERE vt.voto = 'Abstenção')  AS abstencoes,
                COUNT(*)                                        AS total_votacoes
            FROM votos vt
            JOIN deputados d   ON d.id = vt.deputado_id
            JOIN votacoes v    ON v.id = vt.votacao_id
            LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
            WHERE
                LOWER(n.tema_inferido) LIKE :tema
                OR n.keywords::text LIKE :tema
            GROUP BY d.id, d.nome_eleitoral, d.partido, d.uf
            ORDER BY total_votacoes DESC
        """)
        rows = session.execute(sql, {"tema": f"%{tema.lower()}%"}).mappings().all()
        return [dict(r) for r in rows]
    finally:
        session.close()
