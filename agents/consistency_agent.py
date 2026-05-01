"""
agents/consistency_agent.py — Speech-vote consistency analysis agent.

Orchestrates the computation of the Speech Fidelity Index (IFD) for all deputies
by combining stance detection, law direction classification, and vote position
inference from core/stance_detector.py.

Workflow:
  1. Load deputies that have both speeches (with topic) and votes in the database
  2. For each deputy, compute IFD via calcular_ifd_deputado()
  3. Persist results in the ifd_deputados table
  4. Expose reports: ranking by IFD, alerts, per-deputy profile

Usage:
    from agents.consistency_agent import ConsistencyAgent

    agent = ConsistencyAgent()
    agent.criar_tabelas()                          # run once
    agent.processar_deputado(204534)               # test with one deputy first
    agent.processar_todos(limite_por_deputado=30)  # all deputies
    agent.relatorio_ifd()                          # consistency ranking
    agent.relatorio_alertas()                      # contradiction alerts
"""

from __future__ import annotations
import json
import time
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.stance_detector import calcular_ifd_deputado
from core.database import get_session as db_session


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def _salvar_ifd(resultado: dict):
    """Persists or updates the IFD result for a deputy."""
    dep_id      = resultado["deputado_id"]
    alertas_json = json.dumps(resultado.get("alertas", []), ensure_ascii=False)

    session = db_session()
    try:
        existe = session.execute(
            text("SELECT deputado_id FROM ifd_deputados WHERE deputado_id = :id"),
            {"id": dep_id}
        ).fetchone()

        if existe:
            session.execute(text("""
                UPDATE ifd_deputados SET
                    ifd           = :ifd,
                    consistentes  = :cons,
                    total_validos = :total,
                    alertas       = CAST(:alertas AS jsonb),
                    atualizado_em = NOW()
                WHERE deputado_id = :id
            """), {
                "ifd":     resultado["ifd"],
                "cons":    resultado["consistentes"],
                "total":   resultado["total_validos"],
                "alertas": alertas_json,
                "id":      dep_id,
            })
        else:
            session.execute(text("""
                INSERT INTO ifd_deputados
                    (deputado_id, ifd, consistentes, total_validos, alertas, atualizado_em)
                VALUES
                    (:id, :ifd, :cons, :total, CAST(:alertas AS jsonb), NOW())
            """), {
                "id":      dep_id,
                "ifd":     resultado["ifd"],
                "cons":    resultado["consistentes"],
                "total":   resultado["total_validos"],
                "alertas": alertas_json,
            })

        session.commit()
        logger.debug(f"IFD saved for deputy {dep_id} — score={resultado['ifd']}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error saving IFD for deputy {dep_id}: {e}")
    finally:
        session.close()


def _listar_deputados_elegiveis() -> list[int]:
    """
    Returns IDs of deputies that have both speeches (with topic assigned)
    and votes in the database — the minimum requirement for IFD computation.
    """
    session = db_session()
    try:
        rows = session.execute(text("""
            SELECT DISTINCT d.id_deputado
            FROM discursos d
            JOIN votos vt       ON vt.deputado_id = d.id_deputado
            JOIN votacoes_nlp n  ON n.votacao_id   = vt.votacao_id
            WHERE
                d.category_final IS NOT NULL
                AND d.category_final NOT IN ('uncategorized', 'outros')   
                AND n.tema_inferido   IS NOT NULL
                AND n.tema_inferido  NOT IN ('uncategorized', 'outros') 
                AND d.category_final   = n.tema_inferido
            ORDER BY d.id_deputado
        """)).fetchall()
        return [r[0] for r in rows]
    finally:
        session.close()


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

class ConsistencyAgent:
    """
    Computes and stores the Speech Fidelity Index (IFD) for all deputies.

    Recommended workflow:
        agent = ConsistencyAgent()
        agent.criar_tabelas()
        agent.processar_deputado(204534)               # test first
        agent.processar_todos(limite_por_deputado=30)
        agent.relatorio_ifd()
        agent.relatorio_alertas()
    """

    def criar_tabelas(self):
        """Creates the ifd_deputados table. Run once before first use."""
        session = db_session()
        try:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS ifd_deputados (
                    deputado_id   INTEGER PRIMARY KEY REFERENCES deputados(id),
                    ifd           FLOAT,
                    consistentes  INTEGER   DEFAULT 0,
                    total_validos INTEGER   DEFAULT 0,
                    alertas       JSONB     DEFAULT '[]',
                    atualizado_em TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_ifd_score
                    ON ifd_deputados (ifd);
            """))
            session.commit()
            logger.info("Table 'ifd_deputados' ready.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating tables: {e}")
        finally:
            session.close()

    def processar_todos(
        self,
        limite_por_deputado: int   = 30,
        delay:               float = 0.5,
    ) -> int:
        """
        Computes IFD for all eligible deputies.

        Args:
            limite_por_deputado: Max speech-vote pairs per deputy.
                                 30 pairs × ~2 Ollama calls × ~2s = ~2 min per deputy.
            delay:               Seconds to wait between deputies.

        Returns:
            Number of deputies successfully processed.
        """
        ids = _listar_deputados_elegiveis()

        if not ids:
            logger.warning("No eligible deputies found. Check that discursos.category_final is populated.")
            return 0

        logger.info(f"Computing IFD for {len(ids)} deputies...")
        processados = 0

        for dep_id in tqdm(ids, desc="IFD"):
            try:
                resultado = calcular_ifd_deputado(
                    deputado_id=dep_id,
                    limite_pares=limite_por_deputado,
                )
                if resultado["total_validos"] > 0:
                    _salvar_ifd(resultado)
                    processados += 1
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error processing deputy {dep_id}: {e}")

        logger.info(f"IFD computed for {processados}/{len(ids)} deputies.")
        return processados

    def processar_deputado(self, deputado_id: int, limite: int = 50) -> dict:
        """Computes and saves IFD for a single deputy. Useful for testing."""
        resultado = calcular_ifd_deputado(deputado_id, limite_pares=limite)
        if resultado["total_validos"] > 0:
            _salvar_ifd(resultado)
        return resultado

    def relatorio_ifd(self, limit: int = 50):
        """
        Prints a ranking of deputies sorted by IFD score (lowest first).
        Low IFD = frequent contradiction between speeches and votes.
        """
        session = db_session()
        try:
            rows = session.execute(text("""
                SELECT
                    d.nome_eleitoral,
                    d.partido,
                    d.uf,
                    i.ifd,
                    i.consistentes,
                    i.total_validos,
                    jsonb_array_length(i.alertas) AS n_alertas
                FROM ifd_deputados i
                JOIN deputados d ON d.id = i.deputado_id
                WHERE i.ifd IS NOT NULL
                ORDER BY i.ifd ASC
                LIMIT :lim
            """), {"lim": limit}).mappings().all()
        finally:
            session.close()

        if not rows:
            print("No IFD data found. Run processar_todos() first.")
            return

        print(f"\n{'='*78}")
        print(f"{'DEPUTY':<30} {'PARTY':<8} {'UF':<4} {'IFD':>6} {'CONSISTENT':>11} {'TOTAL':>7} {'ALERTS':>7}")
        print(f"{'='*78}")
        for r in rows:
            ifd_str = f"{r['ifd']:.0%}" if r["ifd"] is not None else "N/A"
            print(
                f"{r['nome_eleitoral']:<30} "
                f"{r['partido']:<8} "
                f"{r['uf']:<4} "
                f"{ifd_str:>6} "
                f"{r['consistentes']:>11} "
                f"{r['total_validos']:>7} "
                f"{r['n_alertas']:>7}"
            )
        print(f"{'='*78}\n")

    def relatorio_alertas(self, min_alertas: int = 1):
        """
        Prints all high-confidence speech-vote contradictions.
        These are cases where a deputy clearly argued for something in speeches
        but voted against it — or vice versa — with high LLM confidence.
        """
        session = db_session()
        try:
            rows = session.execute(text("""
                SELECT
                    d.nome_eleitoral,
                    d.partido,
                    d.uf,
                    i.ifd,
                    i.alertas
                FROM ifd_deputados i
                JOIN deputados d ON d.id = i.deputado_id
                WHERE jsonb_array_length(i.alertas) >= :min_a
                ORDER BY jsonb_array_length(i.alertas) DESC
            """), {"min_a": min_alertas}).mappings().all()
        finally:
            session.close()

        if not rows:
            print("No alerts found.")
            return

        print(f"\n{'='*70}")
        print("HIGH-CONFIDENCE SPEECH-VOTE CONTRADICTIONS")
        print(f"{'='*70}")

        for r in rows:
            alertas = r["alertas"]
            if isinstance(alertas, str):
                alertas = json.loads(alertas)

            print(f"\n{r['nome_eleitoral']} ({r['partido']}/{r['uf']}) — IFD: {r['ifd']:.0%}")
            for i, a in enumerate(alertas, 1):
                print(f"  [{i}] Topic:         {a.get('tema')}")
                print(f"      Speech stance: {a.get('stance_discurso')} — {a.get('justificativa')}")
                print(f"      Law direction: {a.get('direcao_lei')}")
                print(f"      Vote:          {a.get('voto')} → revealed: {a.get('posicao_voto')}")
                print(f"      Law summary:   {a.get('ementa')}")

        print(f"\n{'='*70}\n")

    def perfil_deputado(self, deputado_id: int):
        """Prints the full IFD profile for a single deputy."""
        session = db_session()
        try:
            dep = session.execute(text("""
                SELECT nome_eleitoral, partido, uf FROM deputados WHERE id = :id
            """), {"id": deputado_id}).mappings().fetchone()

            ifd_row = session.execute(text("""
                SELECT * FROM ifd_deputados WHERE deputado_id = :id
            """), {"id": deputado_id}).mappings().fetchone()

        finally:
            session.close()

        if not dep:
            print(f"Deputy {deputado_id} not found.")
            return

        print(f"\n{'='*60}")
        print(f"IFD PROFILE: {dep['nome_eleitoral']} ({dep['partido']}/{dep['uf']})")
        print(f"{'='*60}")

        if not ifd_row or ifd_row["ifd"] is None:
            print("No IFD data available. Run processar_deputado() first.")
            return

        ifd   = ifd_row["ifd"]
        label = (
            "HIGH — speeches consistently match votes"    if ifd >= 0.75 else
            "MEDIUM — occasional divergence detected"    if ifd >= 0.50 else
            "LOW — frequent contradiction detected"
        )

        print(f"\n  IFD Score:    {ifd:.1%}")
        print(f"  Assessment:   {label}")
        print(f"  Comparisons:  {ifd_row['consistentes']} consistent / {ifd_row['total_validos']} total")

        alertas = ifd_row["alertas"]
        if isinstance(alertas, str):
            alertas = json.loads(alertas)

        if alertas:
            print(f"\n  Contradictions ({len(alertas)}):")
            for a in alertas:
                print(f"    - [{a.get('tema')}]")
                print(f"      Said {a.get('stance_discurso')}: {a.get('justificativa')}")
                print(f"      Voted {a.get('voto')} on a {a.get('direcao_lei')} law → {a.get('posicao_voto')}")
        else:
            print("\n  No high-confidence contradictions detected.")

        print(f"{'='*60}\n")


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    agent = ConsistencyAgent()
    agent.criar_tabelas()

    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        # Single deputy: python -m agents.consistency_agent 204534
        agent.processar_deputado(int(sys.argv[1]))
        agent.perfil_deputado(int(sys.argv[1]))
    else:
        limite = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        agent.processar_todos(limite_por_deputado=limite)
        agent.relatorio_ifd()
        agent.relatorio_alertas()
