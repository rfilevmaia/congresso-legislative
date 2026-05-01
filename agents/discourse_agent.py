"""
agents/discourse_agent.py — Importação e comparação discurso × votação.

Etapas:
  1. ImportAgent  — lê o parquet e persiste discursos no banco
  2. ResumoAgent  — gera resumo via Qwen3.5 para discursos sem resumo
  3. CompareAgent — cruza keywords de discursos com votações e calcula coerência

Lógica de coerência:
  Sentimento do discurso (POS/NEG/NEU) sobre um tema
  × voto do deputado em votações do mesmo tema
  = score de coerência (-1.0 a +1.0)

  Exemplos:
    discurso POS sobre saúde + voto "Sim" em votação de saúde  → +1.0 (coerente)
    discurso POS sobre saúde + voto "Não" em votação de saúde  → -1.0 (incoerente)
    discurso NEG sobre reforma + voto "Não"                     → +1.0 (coerente)
    sentimento NEU ou overlap < threshold                       →  0.0 (neutro)
"""

from __future__ import annotations
import ast
import json
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.database import get_session as db_session
from core.nlp_local import gerar_resumo
from models.schema import Discurso, DiscursoVotacaoComparacao, Deputado

# Janela temporal: discurso e votação considerados relacionados se
# a diferença for de até JANELA_DIAS (discurso pode ser antes ou depois)
JANELA_DIAS = 90

# Threshold mínimo de overlap para considerar que discurso e votação
# tratam do mesmo tema (jaccard >= MIN_OVERLAP)
MIN_OVERLAP = 0.05


# -- Utilitários --------------------------------------------------------------

def _parse_lista(valor) -> list:
    """Converte string/lista de keywords para list[str] normalizada."""
    if valor is None:
        return []
    if isinstance(valor, list):
        return [str(k).strip().lower() for k in valor if k]
    if isinstance(valor, str):
        val = valor.strip()
        if not val:
            return []
        # Tenta JSON/Python literal primeiro
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(k).strip().lower() for k in parsed if k]
        except Exception:
            pass
        # Fallback: separado por vírgula
        return [k.strip().lower() for k in val.split(",") if k.strip()]
    return []


def _jaccard(a: list, b: list) -> float:
    """Score de Jaccard entre duas listas de keywords."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _calcular_coerencia(
    sentimento: str,
    confidence: float,
    voto: str,
    score_overlap: float,
) -> tuple[str, float]:
    """
    Retorna (label_coerencia, score_coerencia).

    Score:
      +1.0 = totalmente coerente
       0.0 = neutro / indefinido
      -1.0 = totalmente incoerente

    Lógica:
      POS + Sim  → coerente  (+1)
      POS + Nao  → incoerente (-1)
      NEG + Nao  → coerente  (+1)
      NEG + Sim  → incoerente (-1)
      NEU        → neutro    (0)
      Abstencao / Obstucao → neutro (0)
    """
    if score_overlap < MIN_OVERLAP:
        return "indefinido", 0.0

    if sentimento == "NEU" or voto in ("Abstenção", "Obstrução", "Art. 17", "Presidente"):
        return "neutro", 0.0

    voto_favoravel = voto == "Sim"
    discurso_favoravel = sentimento == "POS"

    if voto_favoravel == discurso_favoravel:
        label = "coerente"
        score = round(confidence * score_overlap, 4)
    else:
        label = "incoerente"
        score = round(-confidence * score_overlap, 4)

    return label, score


# -- 1. Importação do parquet -------------------------------------------------

class ImportAgent:
    """
    Lê o arquivo parquet e persiste os discursos na tabela discursos.
    Idempotente: pula registros já existentes (mesmo id_deputado + data).
    """

    def importar(self, caminho_parquet: str) -> int:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas nao instalado. Execute: pip install pandas pyarrow")

        df = pd.read_parquet(caminho_parquet)
        logger.info(f"Parquet carregado: {len(df)} discursos, colunas: {list(df.columns)}")

        inseridos = 0
        pulados = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Importando discursos"):
            try:
                id_dep = int(row["id_deputado"])
                data_str = str(row.get("data", "")).strip()
                try:
                    data = datetime.strptime(data_str[:10], "%Y-%m-%d")
                except Exception:
                    data = None

                session = db_session()
                try:
                    # Verifica se já existe
                    existe = session.execute(
                        text("""
                            SELECT id FROM discursos
                            WHERE id_deputado = :dep AND data = :dt
                        """),
                        {"dep": id_dep, "dt": data}
                    ).fetchone()

                    if existe:
                        pulados += 1
                        continue

                    # Garante que o deputado existe no banco
                    dep = session.get(Deputado, id_dep)
                    if not dep:
                        dep = Deputado(
                            id=id_dep,
                            nome_civil=f"Deputado {id_dep}",
                            nome_eleitoral=f"Deputado {id_dep}",
                        )
                        session.add(dep)
                        session.flush()

                    kw_api   = _parse_lista(row.get("keywords"))
                    kw_tfidf = _parse_lista(row.get("tfidf_keywords"))

                    disc = Discurso(
                        id_deputado      = id_dep,
                        data             = data,
                        keywords_api     = kw_api,
                        keywords_tfidf   = kw_tfidf,
                        transcricao      = str(row.get("transcricao", "") or "")[:50000],
                        transcricao_limpa= str(row.get("transcription_clean", "") or "")[:50000],
                        local            = str(row.get("local", "") or "")[:100],
                        label            = str(row.get("label", "") or "")[:10],
                        label_emotion    = str(row.get("label_emotion", "") or "")[:50],
                        confidence       = float(row.get("confidence", 0) or 0),
                        resumo           = None,
                        resumo_gerado    = False,
                    )
                    session.add(disc)
                    session.commit()
                    inseridos += 1

                except Exception as e:
                    session.rollback()
                    logger.error(f"Erro ao inserir discurso dep={id_dep} data={data_str}: {e}")
                finally:
                    session.close()

            except Exception as e:
                logger.error(f"Erro ao processar linha: {e}")

        logger.info(f"Importacao concluida: {inseridos} inseridos, {pulados} ja existiam.")
        return inseridos


# -- 2. Geração de resumos ----------------------------------------------------

class ResumoDiscursoAgent:
    """
    Gera resumos dos discursos via Qwen3.5 local para os que ainda não têm.
    Usa transcricao_limpa como input — texto sem cabeçalhos protocolares.
    """

    def gerar_pendentes(self, limite: int = 200) -> int:
        session = db_session()
        try:
            ids = [row[0] for row in session.execute(text("""
                SELECT id FROM discursos
                WHERE resumo_gerado = FALSE OR resumo IS NULL
                ORDER BY data DESC
                LIMIT :lim
            """), {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("Nenhum discurso pendente de resumo.")
            return 0

        logger.info(f"Gerando resumos para {len(ids)} discursos...")
        gerados = 0

        for disc_id in tqdm(ids, desc="Resumos"):
            try:
                if self._gerar_um(disc_id):
                    gerados += 1
            except Exception as e:
                logger.error(f"Erro no resumo do discurso {disc_id}: {e}")

        logger.info(f"Resumos gerados: {gerados}/{len(ids)}")
        return gerados

    def _gerar_um(self, disc_id: int) -> bool:
        session = db_session()
        try:
            disc = session.get(Discurso, disc_id)
            if not disc:
                return False
            texto = disc.transcricao_limpa or disc.transcricao or ""
        finally:
            session.close()

        if len(texto.strip()) < 50:
            return False

        prompt_contexto = f"Sentimento do discurso: {disc.label} (confiança: {disc.confidence:.0%})"
        resumo = gerar_resumo(
            texto=texto[:4000],
            contexto_extra=prompt_contexto,
        )

        session = db_session()
        try:
            disc = session.get(Discurso, disc_id)
            if disc:
                disc.resumo = resumo
                disc.resumo_gerado = True
                session.commit()
                return True
        except Exception as e:
            session.rollback()
            logger.error(f"Erro ao salvar resumo do discurso {disc_id}: {e}")
        finally:
            session.close()
        return False


# -- 3. Comparação discurso × votação -----------------------------------------

class CompareAgent:
    """
    Cruza keywords dos discursos com keywords das votações e calcula coerência.

    Para cada discurso de um deputado, busca votações do mesmo deputado
    dentro da janela temporal JANELA_DIAS e calcula o score de overlap
    e coerência para cada par (discurso, votação).
    """

    def comparar_todos(self, limite_discursos: int = 500) -> int:
        """Processa discursos que ainda não têm comparações calculadas."""
        session = db_session()
        try:
            ids = [row[0] for row in session.execute(text("""
                SELECT d.id
                FROM discursos d
                WHERE NOT EXISTS (
                    SELECT 1 FROM discurso_votacao_comparacoes c
                    WHERE c.discurso_id = d.id
                )
                AND d.data IS NOT NULL
                ORDER BY d.data DESC
                LIMIT :lim
            """), {"lim": limite_discursos})]
        finally:
            session.close()

        if not ids:
            logger.info("Nenhum discurso pendente de comparacao.")
            return 0

        logger.info(f"Comparando {len(ids)} discursos com votacoes...")
        total_pares = 0
        for disc_id in tqdm(ids, desc="Comparando"):
            try:
                total_pares += self._comparar_um(disc_id)
            except Exception as e:
                logger.error(f"Erro na comparacao do discurso {disc_id}: {e}")

        logger.info(f"Total de pares discurso x votacao gerados: {total_pares}")
        return total_pares

    def comparar_deputado(self, id_deputado: int) -> int:
        """Recalcula todas as comparações de um deputado específico."""
        session = db_session()
        try:
            ids = [row[0] for row in session.execute(
                text("SELECT id FROM discursos WHERE id_deputado = :dep ORDER BY data DESC"),
                {"dep": id_deputado}
            )]
        finally:
            session.close()
        total = sum(self._comparar_um(i, forcar=True) for i in ids)
        logger.info(f"Deputado {id_deputado}: {total} pares comparados.")
        return total

    def _comparar_um(self, disc_id: int, forcar: bool = False) -> int:
        """Compara um discurso com todas as votações na janela temporal."""
        session = db_session()
        try:
            disc = session.get(Discurso, disc_id)
            if not disc or not disc.data:
                return 0

            id_dep   = disc.id_deputado
            data_disc = disc.data
            kw_disc  = list(set(
                (disc.keywords_tfidf or []) + (disc.keywords_api or [])
            ))
            sentimento  = disc.label or "NEU"
            confidence  = disc.confidence or 0.0

            # Busca votações do deputado na janela temporal
            dt_min = data_disc - timedelta(days=JANELA_DIAS)
            dt_max = data_disc + timedelta(days=JANELA_DIAS)

            rows = session.execute(text("""
                SELECT
                    vt.votacao_id,
                    vt.voto,
                    n.keywords,
                    v.data
                FROM votos vt
                JOIN votacoes v    ON v.id = vt.votacao_id
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                WHERE
                    vt.deputado_id = :dep
                    AND v.data BETWEEN :dt_min AND :dt_max
                    AND n.keywords IS NOT NULL
            """), {"dep": id_dep, "dt_min": dt_min, "dt_max": dt_max}).fetchall()

        finally:
            session.close()

        inseridos = 0
        for votacao_id, voto, kw_json, data_vot in rows:
            try:
                kw_vot = kw_json if isinstance(kw_json, list) else []
                kw_comuns = list(set(kw_disc) & set(kw_vot))
                score_overlap = _jaccard(kw_disc, kw_vot)
                dias_dif = (data_disc - data_vot).days if data_vot else 0
                coerencia, score_coe = _calcular_coerencia(
                    sentimento, confidence, voto or "", score_overlap
                )

                session2 = db_session()
                try:
                    # Upsert
                    existe = session2.execute(text("""
                        SELECT id FROM discurso_votacao_comparacoes
                        WHERE discurso_id = :d AND votacao_id = :v
                    """), {"d": disc_id, "v": votacao_id}).fetchone()

                    if existe and not forcar:
                        continue

                    if existe:
                        session2.execute(text("""
                            UPDATE discurso_votacao_comparacoes
                            SET keywords_comuns=:kc, n_keywords_comuns=:nkc,
                                score_overlap=:so, voto=:vt, sentimento_discurso=:sd,
                                confidence=:cf, coerencia=:co, score_coerencia=:sc,
                                dias_diferenca=:dd
                            WHERE discurso_id=:d AND votacao_id=:v
                        """), {
                            "kc": json.dumps(kw_comuns), "nkc": len(kw_comuns),
                            "so": score_overlap, "vt": voto, "sd": sentimento,
                            "cf": confidence, "co": coerencia, "sc": score_coe,
                            "dd": dias_dif, "d": disc_id, "v": votacao_id
                        })
                    else:
                        comp = DiscursoVotacaoComparacao(
                            discurso_id         = disc_id,
                            votacao_id          = votacao_id,
                            id_deputado         = id_dep,
                            keywords_comuns     = kw_comuns,
                            n_keywords_comuns   = len(kw_comuns),
                            score_overlap       = score_overlap,
                            voto                = voto,
                            sentimento_discurso = sentimento,
                            confidence          = confidence,
                            coerencia           = coerencia,
                            score_coerencia     = score_coe,
                            dias_diferenca      = dias_dif,
                        )
                        session2.add(comp)

                    session2.commit()
                    inseridos += 1

                except Exception as e:
                    session2.rollback()
                    logger.error(f"Erro ao salvar comparacao disc={disc_id} vot={votacao_id}: {e}")
                finally:
                    session2.close()

            except Exception as e:
                logger.error(f"Erro ao processar par ({disc_id}, {votacao_id}): {e}")

        return inseridos


# -- 4. Consultas analíticas --------------------------------------------------

def relatorio_coerencia_deputado(id_deputado: int, min_overlap: float = 0.05):
    """
    Mostra o perfil de coerência de um deputado:
    quantas vezes foi coerente/incoerente por tema.
    """
    session = db_session()
    try:
        rows = session.execute(text("""
            SELECT
                n.tema_inferido                     AS tema,
                c.coerencia,
                COUNT(*)                            AS total,
                ROUND(AVG(c.score_coerencia)::numeric, 3) AS score_medio,
                ROUND(AVG(c.score_overlap)::numeric, 3)   AS overlap_medio
            FROM discurso_votacao_comparacoes c
            JOIN votacoes_nlp n ON n.votacao_id = c.votacao_id
            WHERE
                c.id_deputado = :dep
                AND c.score_overlap >= :min_ov
                AND c.coerencia IN ('coerente', 'incoerente')
            GROUP BY tema, c.coerencia
            ORDER BY tema, c.coerencia
        """), {"dep": id_deputado, "min_ov": min_overlap}).fetchall()
    finally:
        session.close()

    if not rows:
        print(f"Sem comparacoes suficientes para deputado {id_deputado}.")
        return

    print(f"\nCoerencia discurso x voto — Deputado {id_deputado}")
    print(f"{'TEMA':<25} {'COERENCIA':<12} {'TOTAL':>6} {'SCORE':>7} {'OVERLAP':>8}")
    print("-" * 62)
    for tema, coe, total, score, overlap in rows:
        print(f"{(tema or 'outros'):<25} {coe:<12} {total:>6} {score:>7} {overlap:>8}")
    print()


def ranking_incoerencia(limite: int = 20, min_comparacoes: int = 5):
    """
    Lista os deputados mais incoerentes (discurso x voto).
    """
    session = db_session()
    try:
        rows = session.execute(text("""
            SELECT
                d.nome_eleitoral,
                d.partido,
                d.uf,
                COUNT(*) FILTER (WHERE c.coerencia = 'incoerente')   AS n_incoerentes,
                COUNT(*) FILTER (WHERE c.coerencia = 'coerente')     AS n_coerentes,
                COUNT(*)                                              AS total,
                ROUND(
                    COUNT(*) FILTER (WHERE c.coerencia = 'incoerente') * 100.0 / COUNT(*),
                    1
                )                                                     AS pct_incoerente,
                ROUND(AVG(c.score_coerencia)::numeric, 3)            AS score_medio
            FROM discurso_votacao_comparacoes c
            JOIN deputados d ON d.id = c.id_deputado
            WHERE c.coerencia IN ('coerente', 'incoerente')
            GROUP BY d.id, d.nome_eleitoral, d.partido, d.uf
            HAVING COUNT(*) >= :min_c
            ORDER BY pct_incoerente DESC
            LIMIT :lim
        """), {"min_c": min_comparacoes, "lim": limite}).fetchall()
    finally:
        session.close()

    print(f"\nRanking de incoerencia discurso x voto (min {min_comparacoes} comparacoes)")
    print(f"{'DEPUTADO':<30} {'PT':>4} {'UF':>3} {'INCOER':>7} {'COER':>6} {'%INCO':>7} {'SCORE':>7}")
    print("-" * 70)
    for nome, partido, uf, ni, nc, tot, pct, score in rows:
        print(f"{(nome or ''):<30} {(partido or ''):>4} {(uf or ''):>3} {ni:>7} {nc:>6} {pct:>6}% {score:>7}")
    print()


def buscar_incoerencias(id_deputado: int = None, tema: str = None, limite: int = 50):
    """
    Retorna lista de pares (discurso, votação) onde o deputado foi incoerente.
    """
    session = db_session()
    try:
        filtros = ["c.coerencia = 'incoerente'"]
        params = {"lim": limite}
        if id_deputado:
            filtros.append("c.id_deputado = :dep")
            params["dep"] = id_deputado
        if tema:
            filtros.append("LOWER(n.tema_inferido) LIKE :tema")
            params["tema"] = f"%{tema.lower()}%"

        where = " AND ".join(filtros)
        rows = session.execute(text(f"""
            SELECT
                d_dep.nome_eleitoral,
                d_dep.partido,
                d_dep.uf,
                disc.data                   AS data_discurso,
                disc.keywords_tfidf         AS kw_discurso,
                disc.resumo                 AS resumo_discurso,
                disc.label                  AS sentimento,
                v.data                      AS data_votacao,
                v.descricao                 AS descricao_votacao,
                n.tema_inferido             AS tema,
                c.voto,
                c.keywords_comuns,
                c.score_overlap,
                c.score_coerencia,
                c.dias_diferenca
            FROM discurso_votacao_comparacoes c
            JOIN discursos disc        ON disc.id  = c.discurso_id
            JOIN votacoes v            ON v.id     = c.votacao_id
            JOIN deputados d_dep       ON d_dep.id = c.id_deputado
            LEFT JOIN votacoes_nlp n   ON n.votacao_id = v.id
            WHERE {where}
            ORDER BY ABS(c.score_coerencia) DESC
            LIMIT :lim
        """), params).mappings().fetchall()
    finally:
        session.close()

    return [dict(r) for r in rows]


# -- Orquestrador -------------------------------------------------------------

class DiscourseAgent:
    """
    Agente principal que orquestra todas as etapas.

    Uso completo:
        agent = DiscourseAgent()
        agent.importar("caminho/para/saida.parquet")
        agent.gerar_resumos(limite=200)
        agent.comparar(limite=500)
        agent.relatorio(id_deputado=220686)
    """

    def __init__(self):
        self.importer = ImportAgent()
        self.resumo   = ResumoDiscursoAgent()
        self.compare  = CompareAgent()

    def importar(self, caminho_parquet: str) -> int:
        return self.importer.importar(caminho_parquet)

    def gerar_resumos(self, limite: int = 200) -> int:
        return self.resumo.gerar_pendentes(limite)

    def comparar(self, limite: int = 500) -> int:
        return self.compare.comparar_todos(limite)

    def relatorio(self, id_deputado: int = None):
        if id_deputado:
            relatorio_coerencia_deputado(id_deputado)
        else:
            ranking_incoerencia()

    def incoerencias(self, id_deputado: int = None, tema: str = None) -> list:
        return buscar_incoerencias(id_deputado=id_deputado, tema=tema)

    def pipeline_completo(self, caminho_parquet: str):
        """Roda todas as etapas em sequência."""
        logger.info("=== Etapa 1: Importando discursos ===")
        self.importar(caminho_parquet)

        logger.info("=== Etapa 2: Gerando resumos ===")
        self.gerar_resumos()

        logger.info("=== Etapa 3: Comparando discursos x votacoes ===")
        self.comparar()

        logger.info("=== Relatorio final ===")
        ranking_incoerencia()


if __name__ == "__main__":
    import sys
    agent = DiscourseAgent()

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".parquet"):
        agent.pipeline_completo(sys.argv[1])
    elif len(sys.argv) >= 2:
        dep_id = int(sys.argv[1])
        relatorio_coerencia_deputado(dep_id)
        incos = buscar_incoerencias(id_deputado=dep_id, limite=10)
        for i in incos:
            print(f"\n[{i['data_discurso']}] DISCURSO {i['sentimento']} sobre: {i['kw_discurso']}")
            print(f"[{i['data_votacao']}] VOTOU '{i['voto']}' em: {i['descricao_votacao'][:80]}")
            print(f"  Tema: {i['tema']} | Overlap: {i['score_overlap']:.2f} | Score: {i['score_coerencia']:.2f}")
            print(f"  Keywords comuns: {i['keywords_comuns']}")
    else:
        print("Uso:")
        print("  python -m agents.discourse_agent arquivo.parquet   # pipeline completo")
        print("  python -m agents.discourse_agent 220686            # relatorio de um deputado")
