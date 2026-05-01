"""
core/api_camara.py — Funções puras para a API REST da Câmara dos Deputados.

Cada função é autônoma e retorna dicts Python prontos para persistência.
Documentação oficial: https://dadosabertos.camara.leg.br/swagger/api.html

Endpoints utilizados:
  GET /votacoes                    → lista votações por período
  GET /votacoes/{id}               → detalhes de uma votação
  GET /votacoes/{id}/votos         → votos nominais de cada deputado
  GET /votacoes/{id}/orientacoes   → orientações dos partidos (bônus)
  GET /deputados                   → lista de deputados
  GET /deputados/{id}              → perfil completo de um deputado
  GET /proposicoes/{id}            → dados da proposição votada
"""

import time
from datetime import datetime
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
from config import CAMARA_API_BASE, CAMARA_API_TIMEOUT, CAMARA_PAGE_SIZE


# ── Sessão HTTP compartilhada ─────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({
    "Accept": "application/json",
    "User-Agent": "congresso-nlp/1.0 (pesquisa academica)",
})


# ── Utilitário de paginação ───────────────────────────────────────────────────

class VotacaoIndisponivel(Exception):
    """Votação existe na listagem mas não tem detalhes disponíveis na API (404/410)."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get(endpoint: str, params: dict = None) -> dict:
    """GET com retry automático e tratamento de rate-limit (HTTP 429)."""
    url = f"{CAMARA_API_BASE}{endpoint}"
    resp = _session.get(url, params=params, timeout=CAMARA_API_TIMEOUT)

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 5))
        logger.warning(f"Rate limit atingido. Aguardando {retry_after}s...")
        time.sleep(retry_after)
        resp = _session.get(url, params=params, timeout=CAMARA_API_TIMEOUT)

    if resp.status_code in (404, 410):
        # Votação listada mas sem detalhes — comum na API da Câmara
        # Não faz retry, apenas sinaliza para o caller pular
        raise VotacaoIndisponivel(f"HTTP {resp.status_code} em {endpoint}")

    if not resp.ok:
        logger.error(f"HTTP {resp.status_code} em {url} — body: {resp.text[:200]}")

    resp.raise_for_status()
    return resp.json()


def _paginar(endpoint: str, params: dict = None, max_paginas: int = 50) -> list:
    """Percorre todas as páginas de um endpoint paginado e retorna todos os itens."""
    params = params or {}
    params["itens"] = CAMARA_PAGE_SIZE
    todos = []

    for pagina in range(1, max_paginas + 1):
        params["pagina"] = pagina
        data = _get(endpoint, params)
        items = data.get("dados", [])
        if not items:
            break
        todos.extend(items)
        logger.debug(f"{endpoint} — página {pagina} → {len(items)} itens")
        if len(items) < CAMARA_PAGE_SIZE:
            break
        time.sleep(0.3)   # cortesia para não sobrecarregar o servidor

    return todos


# ── Deputados ─────────────────────────────────────────────────────────────────

def listar_deputados(legislatura: Optional[int] = None) -> list[dict]:
    """
    Retorna todos os deputados de uma legislatura (padrão: atual).
    Cada item contém: id, nome, siglaPartido, siglaUf, uri, urlFoto, idLegislatura
    """
    params = {}
    if legislatura:
        params["idLegislatura"] = legislatura
    params["ordem"] = "ASC"
    params["ordenarPor"] = "nome"
    return _paginar("/deputados", params)


def obter_deputado(id_deputado: int) -> dict:
    """
    Retorna o perfil completo de um deputado.
    Inclui: nomeCivil, cpf, dataNascimento, escolaridade, profissões, etc.
    """
    data = _get(f"/deputados/{id_deputado}")
    return data.get("dados", {})


# ── Votações ──────────────────────────────────────────────────────────────────

def listar_votacoes(
    data_inicio: str,
    data_fim: str,
    id_proposicao: Optional[int] = None,
    id_orgao: Optional[int] = None,
) -> list[dict]:
    """
    Lista votações em um período. Formato de data: "YYYY-MM-DD".

    Campos retornados por item:
      id, uri, data, descricao, aprovacao,
      proposicaoObjeto (uri + descricao), placar (sim/não/abstencao)
    """
    params = {
        "dataInicio": data_inicio,
        "dataFim": data_fim,
        "ordenarPor": "dataHoraRegistro",
        "ordem": "DESC",
    }
    if id_proposicao:
        params["idProposicao"] = id_proposicao
    if id_orgao:
        params["idOrgao"] = id_orgao

    return _paginar("/votacoes", params)


def obter_votacao(id_votacao: str) -> dict:
    """
    Retorna os detalhes completos de uma votação, incluindo placar e
    o campo 'aprovacao' (bool) que indica se a proposição foi aprovada.
    """
    data = _get(f"/votacoes/{id_votacao}")
    return data.get("dados", {})


def obter_votos_votacao(id_votacao: str) -> list[dict]:
    """
    Retorna os votos nominais de todos os deputados em uma votação.

    ATENÇÃO: o endpoint /votos NÃO aceita paginação (pagina/itens).
    A API retorna todos os votos de uma vez em dados[].

    Campos por voto:
      tipoVoto  → "Sim" | "Não" | "Abstenção" | "Obstrução" | "Art. 17"
      dataRegistroVoto
      deputado_  → {id, nome, siglaPartido, siglaUf, uri, urlFoto}
    """
    data = _get(f"/votacoes/{id_votacao}/votos")
    return data.get("dados", [])


def obter_orientacoes_votacao(id_votacao: str) -> list[dict]:
    """
    Retorna as orientações dos líderes partidários para a votação.
    Útil para cruzar se o deputado seguiu ou contrariou a orientação do partido.

    Campos: bancada (nome), orientacao (Sim/Não/Abstenção/Liberado)
    """
    data = _get(f"/votacoes/{id_votacao}/orientacoes")
    return data.get("dados", [])


# ── Proposições ───────────────────────────────────────────────────────────────

def obter_proposicao(id_proposicao: int) -> dict:
    """
    Retorna dados completos de uma proposição.

    Campos úteis:
      siglaTipo, numero, ano, ementa, ementaDetalhada,
      keywords, temas, urlInteiroTeor, statusProposicao
    """
    data = _get(f"/proposicoes/{id_proposicao}")
    return data.get("dados", {})


def extrair_id_proposicao_da_votacao(votacao: dict) -> Optional[int]:
    """
    Extrai o ID numérico da proposição a partir do dict de uma votação.
    Tenta múltiplos campos pois a API retorna em estruturas diferentes.
    """
    # Tentativa 1: campo direto (formato antigo)
    uri = votacao.get("uriProposicaoPrincipal") or votacao.get("proposicaoObjeto", {}).get("uri")
    if uri:
        try:
            return int(uri.rstrip("/").split("/")[-1])
        except (ValueError, AttributeError):
            pass

    # Tentativa 2: objetosPossiveis (formato atual da API)
    objetos = votacao.get("objetosPossiveis", [])
    if objetos:
        primeiro = objetos[0]
        if primeiro.get("id"):
            return int(primeiro["id"])
        uri2 = primeiro.get("uri", "")
        try:
            return int(uri2.rstrip("/").split("/")[-1])
        except (ValueError, AttributeError):
            pass

    return None

# ── Função de conveniência ────────────────────────────────────────────────────

def coletar_votacao_completa(id_votacao: str) -> dict:
    votacao = obter_votacao(id_votacao)
    votos = obter_votos_votacao(id_votacao)
    orientacoes = obter_orientacoes_votacao(id_votacao)

    proposicao = {}
    
    # Tenta extrair proposição de objetosPossiveis primeiro (sem chamada extra à API)
    objetos = votacao.get("objetosPossiveis", [])
    if objetos:
        proposicao = objetos[0]  # já tem id, ementa, siglaTipo, numero, ano
        logger.debug(f"Proposição extraída de objetosPossiveis: {proposicao.get('id')}")
    else:
        # Fallback: busca via API
        id_prop = extrair_id_proposicao_da_votacao(votacao)
        if id_prop:
            try:
                proposicao = obter_proposicao(id_prop)
            except Exception as e:
                logger.warning(f"Não foi possível obter proposição {id_prop}: {e}")

    return {
        "votacao": votacao,
        "votos": votos,
        "proposicao": proposicao,
        "orientacoes": orientacoes,
        "coletado_em": datetime.utcnow().isoformat(),
    }
