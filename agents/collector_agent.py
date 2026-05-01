"""
agents/collector_agent.py — Agente de coleta de dados da API da Câmara.

Responsabilidades:
  1. Buscar votações em um intervalo de datas
  2. Para cada votação: coletar detalhes, votos nominais e proposição associada
  3. Persistir tudo no banco de dados
  4. Retornar lista de IDs coletados para que o NLPAgent processe na sequência

Pode ser executado de forma autônoma ou orquestrado pelo PipelineAgent.
"""

from __future__ import annotations
from loguru import logger
from tqdm import tqdm

from core.api_camara import (
    listar_votacoes,
    coletar_votacao_completa,
    listar_deputados,
    VotacaoIndisponivel,
)
from core.database import (
    upsert_deputado,
    upsert_proposicao,
    upsert_votacao,
    salvar_votos_em_lote,
)


class CollectorAgent:
    """
    Agente responsável pela extração e persistência de dados legislativos.

    Uso standalone:
        agent = CollectorAgent()
        ids = agent.coletar_periodo("2024-03-01", "2024-03-31")

    Uso como parte do pipeline:
        ids = CollectorAgent().coletar_periodo(...)
        NLPAgent().processar_votacoes(ids)
    """

    def coletar_deputados(self, legislatura: int = None) -> int:
        """
        Sincroniza o cadastro de deputados no banco de dados.

        Args:
            legislatura: número da legislatura (ex: 57 = atual). None = atual.

        Returns:
            Número de deputados sincronizados.
        """
        logger.info(f"Sincronizando deputados (legislatura={legislatura or 'atual'})...")
        deputados = listar_deputados(legislatura)

        for dep in tqdm(deputados, desc="Deputados"):
            try:
                upsert_deputado(dep)
            except Exception as e:
                logger.warning(f"Erro ao salvar deputado {dep.get('id')}: {e}")

        logger.info(f"✅ {len(deputados)} deputados sincronizados.")
        return len(deputados)

    def coletar_periodo(
        self,
        data_inicio: str,
        data_fim: str,
        id_proposicao: int = None,
    ) -> list[str]:
        """
        Coleta todas as votações de um período e persiste no banco.

        Args:
            data_inicio:    "YYYY-MM-DD"
            data_fim:       "YYYY-MM-DD"
            id_proposicao:  filtra por proposição específica (opcional)

        Returns:
            Lista de IDs das votações coletadas (para repassar ao NLPAgent).
        """
        logger.info(f"Coletando votações de {data_inicio} a {data_fim}...")

        votacoes_resumidas = listar_votacoes(
            data_inicio=data_inicio,
            data_fim=data_fim,
            id_proposicao=id_proposicao,
        )

        if not votacoes_resumidas:
            logger.warning("Nenhuma votação encontrada no período.")
            return []

        logger.info(f"Encontradas {len(votacoes_resumidas)} votações. Coletando detalhes...")
        ids_coletados = []

        for resumo in tqdm(votacoes_resumidas, desc="Votações"):
            id_vot = str(resumo.get("id", ""))
            if not id_vot:
                continue
            try:
                ids_coletados.append(self._processar_votacao(id_vot))
            except VotacaoIndisponivel as e:
                logger.warning(f"Votação {id_vot} indisponível na API, pulando: {e}")
            except Exception as e:
                logger.error(f"Falha ao processar votação {id_vot}: {e}")

        logger.info(f"✅ {len(ids_coletados)} votações coletadas e persistidas.")
        return ids_coletados

    def coletar_votacao_unica(self, id_votacao: str) -> str:
        """
        Coleta e persiste uma votação específica pelo ID.
        Útil para reprocessamento pontual.
        """
        return self._processar_votacao(id_votacao)

    # ── Internos ──────────────────────────────────────────────────────────────

    def _processar_votacao(self, id_votacao: str) -> str:
        """Coleta, persiste e retorna o ID da votação."""
        dados = coletar_votacao_completa(id_votacao)

        # 1. Proposição
        proposicao_id = None
        if dados["proposicao"]:
            proposicao_id = upsert_proposicao(dados["proposicao"])

        # 2. Votação
        upsert_votacao(dados["votacao"], proposicao_id=proposicao_id)

        # 3. Votos nominais (com upsert de deputados embutido)
        if dados["votos"]:
            salvar_votos_em_lote(id_votacao, dados["votos"])

        logger.debug(f"Votação {id_votacao} → {len(dados['votos'])} votos | prop={proposicao_id}")
        return id_votacao
