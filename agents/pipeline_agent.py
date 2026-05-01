"""
agents/pipeline_agent.py — Orquestrador do pipeline completo.

Sequência de execução:
  1. CollectorAgent  → coleta votações da API e persiste no banco
  2. NLPAgent        → enriquece com resumo + keywords + tema (modelos locais)
  3. Relatório final → imprime estatísticas da execução

Este agente pode ser chamado diretamente (script) ou exposto como
função para um framework de agentes (LangChain, CrewAI, AutoGen, etc.).
"""

from __future__ import annotations
from datetime import datetime
from loguru import logger

from agents.collector_agent import CollectorAgent
from agents.nlp_agent import NLPAgent
from core.database import (
    query_deputado_votacoes,
    query_votacoes_por_tema,
    query_deputados_por_tema,
)


class PipelineAgent:
    """
    Orquestrador principal. Une coleta + NLP em uma única chamada.

    Uso:
        agent = PipelineAgent()

        # Executa pipeline completo para um período
        resultado = agent.run(data_inicio="2024-03-01", data_fim="2024-03-31")

        # Apenas NLP para votações já coletadas
        agent.enriquecer_pendentes()

        # Consultas analíticas
        votos = agent.buscar_deputado("Tabata Amaral", tema="educação")
        ranking = agent.ranking_tema("meio ambiente")
    """

    def __init__(self):
        self.collector = CollectorAgent()
        self.nlp       = NLPAgent()

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(
        self,
        data_inicio: str,
        data_fim: str,
        sincronizar_deputados: bool = False,
        id_proposicao: int = None,
    ) -> dict:
        """
        Executa o pipeline completo:
          1. (Opcional) Sincroniza deputados
          2. Coleta votações no período
          3. Roda NLP em todas as votações coletadas
          4. Retorna relatório de execução

        Args:
            data_inicio:            "YYYY-MM-DD"
            data_fim:               "YYYY-MM-DD"
            sincronizar_deputados:  se True, atualiza cadastro de deputados antes
            id_proposicao:          filtra por proposição específica (opcional)

        Returns:
            dict com estatísticas da execução
        """
        inicio = datetime.utcnow()
        logger.info(f"🚀 Pipeline iniciado: {data_inicio} → {data_fim}")

        # Etapa 0: Deputados
        n_deputados = 0
        if sincronizar_deputados:
            n_deputados = self.collector.coletar_deputados()

        # Etapa 1: Coleta
        ids_votacoes = self.collector.coletar_periodo(
            data_inicio=data_inicio,
            data_fim=data_fim,
            id_proposicao=id_proposicao,
        )

        # Etapa 2: NLP
        n_nlp = self.nlp.processar_votacoes(ids_votacoes)

        duracao = (datetime.utcnow() - inicio).total_seconds()
        relatorio = {
            "periodo": f"{data_inicio} → {data_fim}",
            "deputados_sincronizados": n_deputados,
            "votacoes_coletadas": len(ids_votacoes),
            "votacoes_com_nlp": n_nlp,
            "duracao_segundos": round(duracao, 1),
        }
        logger.info(f"✅ Pipeline concluído em {duracao:.1f}s | {relatorio}")
        return relatorio

    def enriquecer_pendentes(self, limite: int = 500) -> int:
        """
        Roda NLP apenas nas votações ainda não enriquecidas.
        Útil para execução incremental (cron, Lambda agendado).
        """
        return self.nlp.processar_pendentes(limite=limite)

    # ── Consultas analíticas ──────────────────────────────────────────────────

    def buscar_deputado(
        self,
        nome: str = None,
        deputado_id: int = None,
        tema: str = None,
        data_inicio: str = None,
        data_fim: str = None,
    ) -> list[dict]:
        """
        Retorna os votos de um deputado, opcionalmente filtrados por tema.

        Cada item do resultado contém:
          - nome, partido, UF do deputado
          - ID, data, descrição e resultado (aprovação) da votação
          - voto do deputado (Sim/Não/Abstenção/...)
          - tema NLP, keywords e resumo da votação
        """
        return query_deputado_votacoes(
            nome_deputado=nome,
            deputado_id=deputado_id,
            tema_keyword=tema,
            data_inicio=data_inicio,
            data_fim=data_fim,
        )

    def buscar_tema(self, tema: str, limit: int = 50) -> list[dict]:
        """Retorna todas as votações relacionadas a um tema."""
        return query_votacoes_por_tema(tema, limit=limit)

    def ranking_tema(self, tema: str) -> list[dict]:
        """
        Para um tema, retorna o ranking de deputados por posicionamento.
        Mostra quantas vezes cada um votou Sim/Não/Abstenção em votações do tema.
        """
        return query_deputados_por_tema(tema)

    def imprimir_relatorio_deputado(self, nome: str, tema: str = None):
        """Imprime um relatório formatado de votações de um deputado."""
        votos = self.buscar_deputado(nome=nome, tema=tema)
        if not votos:
            print(f"Nenhum resultado para deputado '{nome}'" +
                  (f" no tema '{tema}'" if tema else ""))
            return

        dep = votos[0]
        print(f"\n{'='*60}")
        print(f"Deputado: {dep['nome_deputado']} ({dep['partido']}/{dep['uf']})")
        if tema:
            print(f"Tema filtrado: {tema}")
        print(f"Total de votações: {len(votos)}")
        print(f"{'='*60}\n")

        for v in votos:
            status = "✅ APROVADO" if v["aprovacao"] else "❌ REJEITADO" if v["aprovacao"] is False else "❓"
            voto_emoji = {"Sim": "👍", "Não": "👎", "Abstenção": "🤷"}.get(v["voto"], "❓")
            print(f"[{v['data_votacao']}] {status}")
            print(f"  Votação: {v['votacao_id']}")
            print(f"  Descrição: {(v['descricao'] or '')[:100]}...")
            print(f"  Voto: {voto_emoji} {v['voto']}")
            print(f"  Tema: {v.get('tema_inferido', 'N/A')}")
            if v.get("keywords"):
                print(f"  Keywords: {', '.join(v['keywords'][:5])}")
            if v.get("resumo"):
                print(f"  Resumo: {v['resumo'][:200]}...")
            print()


# ── Execução direta ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    agent = PipelineAgent()

    if len(sys.argv) >= 3:
        # python pipeline_agent.py 2024-03-01 2024-03-31
        resultado = agent.run(
            data_inicio=sys.argv[1],
            data_fim=sys.argv[2],
            sincronizar_deputados="--sync-deputados" in sys.argv,
        )
        print(resultado)
    else:
        print("Uso: python pipeline_agent.py YYYY-MM-DD YYYY-MM-DD [--sync-deputados]")
        print("\nExemplo de consulta:")
        agent.imprimir_relatorio_deputado("Tabata Amaral", tema="educação")
