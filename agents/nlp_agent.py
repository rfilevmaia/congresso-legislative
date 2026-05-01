"""
agents/nlp_agent.py — Agente de enriquecimento semântico das votações.

Responsabilidades:
  1. Ler votações persistidas que ainda não têm análise NLP
  2. Montar o texto-base (ementa + descrição da votação)
  3. Extrair keywords via BERTimbau Large (MPS — Apple Silicon)
  4. Gerar resumo via Qwen2.5 via Ollama (MLX backend)
  5. Inferir tema e salvar em votacoes_nlp

Os modelos são carregados uma única vez (lazy init) e reutilizados
ao longo de toda a sessão de processamento, evitando overhead de I/O.
"""

from __future__ import annotations
from loguru import logger
from tqdm import tqdm
from sqlalchemy import text

from core.nlp_local import processar_votacao_nlp
from core.database import upsert_votacao_nlp, get_session
from models.schema import Votacao, Proposicao, VotacaoNLP

from bertopic import BERTopic #keywork evaluation compatibility

_bertopic_model = None  # lazy init
    
class NLPAgent:
    """
    Agente que roda os modelos NLP locais para enriquecer votações.

    Uso standalone:
        agent = NLPAgent()
        agent.processar_pendentes()          # processa tudo sem NLP ainda
        agent.processar_votacoes([id1, id2]) # processa IDs específicos

    Uso no pipeline:
        NLPAgent().processar_votacoes(ids_retornados_pelo_collector)
    """

    
    # ── Novo: carregamento do BERTopic treinado ───────────────────────────────────

from bertopic import BERTopic

    _bertopic_model = None
    model_path='/Volumes/LaCie/development/congresso/models/'

    TOPIC_TO_CATEGORY = {
        -1: "uncategorized",
         0: "uncategorized",
         1: "direitos sociais",
         2: "política externa",
         3: "direitos civis e liberdade de expressão",
         4: "pessoas com deficiência",
         5: "aborto e direito à vida",
         6: "educação",
         7: "direitos da mulher",
         8: "pandemia e saúde pública",
         9: "cultura",
        10: "educação",
        11: "crise climática",
        12: "religião",
        13: "direitos da mulher",
        14: "uncategorized",
    }

    def _get_bertopic() -> "BERTopic":
        global _bertopic_model
        if _bertopic_model is not None:
            return _bertopic_model

        import pickle

        model_path = "/Volumes/LaCie/development/congresso/models/bertopic_v3.pkl"
        logger.info(f"Loading BERTopic from:: {model_path}...")

        with open(model_path, "rb") as f:
            _bertopic_model = pickle.load(f)

        logger.info("✅ BERTopic loaded.")
        return _bertopic_model


    def _inferir_tema_bertopic(texto: str) -> str:
        if not texto or len(texto.strip()) < 20:
            return "uncategorized"

        CONFIDENCE_THRESHOLD = 0.30

        try:
            model = _get_bertopic()
            topics, probs = model.transform([texto])
            topic_id   = topics[0]
            confidence = float(probs[0].max()) if hasattr(probs[0], "max") else float(probs[0])

            if confidence < CONFIDENCE_THRESHOLD:
                return "uncategorized"

            return TOPIC_TO_CATEGORY.get(topic_id, "uncategorized")

        except Exception as e:
            logger.error(f"BERTopic.transform error: {e}")
            return _inferir_tema_fallback(texto)  # fallback para o método antigo
        
    def _inferir_tema_fallback(keywords: list[str], texto: str) -> str:
        """
        Fallback theme inference using keyword matching.
        Used when BERTopic model is unavailable or returns low confidence.
        """
        TEMAS = {
            "economia": ["fiscal", "orçamento", "imposto", "tributo", "economia", "financeiro",
                         "dívida", "previdência", "benefício", "auxílio", "bolsa"],
            "saúde": ["saúde", "sus", "medicamento", "hospital", "doença", "médico",
                      "sanitário", "epidemia", "vacina", "enfermagem"],
            "educação": ["educação", "escola", "ensino", "universidade", "professor",
                         "aluno", "formação", "letramento", "mec"],
            "segurança pública": ["crime", "polícia", "penal", "prisão", "segurança",
                                  "violência", "arma", "tráfico", "pena", "presídio"],
            "meio ambiente": ["ambiental", "floresta", "desmatamento", "carbono", "clima",
                              "biodiversidade", "resíduo", "saneamento", "água"],
            "direitos humanos": ["direito", "igualdade", "discriminação", "gênero",
                                 "racial", "indígena", "criança", "idoso", "lgbtq"],
            "infraestrutura": ["infraestrutura", "rodovia", "ferrovia", "porto", "aeroporto",
                               "energia", "elétrica", "petróleo", "gás", "combustível"],
            "reforma política": ["eleição", "partido", "candidato", "voto", "urna",
                                 "político", "mandato", "reforma", "constituição"],
            "agricultura": ["agropecuária", "rural", "produtor", "safra", "agrotóxico",
                            "irrigação", "crédito rural", "fundiário"],
        }

        texto_lower = texto.lower()
        todas_palavras = keywords + re.findall(r'\b\w{5,}\b', texto_lower)

        pontuacao = {}
        for tema, termos in TEMAS.items():
            score = sum(1 for t in termos if any(t in p for p in todas_palavras))
            if score > 0:
                pontuacao[tema] = score

        if not pontuacao:
            return "outros"
        return max(pontuacao, key=pontuacao.get)
            
    def processar_votacoes(self, votacao_ids: list[str]) -> int:
        """
        Processa uma lista de IDs de votações, gerando e salvando
        resumo + keywords + tema para cada uma.

        Args:
            votacao_ids: lista de IDs de votação (strings)

        Returns:
            Número de votações processadas com sucesso.
        """
        if not votacao_ids:
            logger.info("Nenhuma votação para processar (NLP).")
            return 0

        logger.info(f"Iniciando NLP para {len(votacao_ids)} votações...")
        processadas = 0

        for id_vot in tqdm(votacao_ids, desc="NLP"):
            try:
                self._processar_uma(id_vot)
                processadas += 1
            except Exception as e:
                logger.error(f"Erro no NLP da votação {id_vot}: {e}")

        logger.info(f"✅ NLP concluído: {processadas}/{len(votacao_ids)} votações.")
        return processadas

    def processar_pendentes(self, limite: int = 500) -> int:
        """
        Busca no banco as votações que ainda não têm análise NLP e as processa.
        Ideal para jobs periódicos (cron, Lambda, etc.).

        Args:
            limite: máximo de votações a processar nesta execução.
        """
        session = get_session()
        try:
            sql = text("""
                SELECT v.id
                FROM votacoes v
                LEFT JOIN votacoes_nlp n ON n.votacao_id = v.id
                WHERE n.id IS NULL
                ORDER BY v.data DESC
                LIMIT :lim
            """)
            ids = [row[0] for row in session.execute(sql, {"lim": limite})]
        finally:
            session.close()

        if not ids:
            logger.info("Nenhuma votação pendente de análise NLP.")
            return 0

        logger.info(f"Encontradas {len(ids)} votações pendentes de NLP.")
        return self.processar_votacoes(ids)

    def reprocessar_votacao(self, votacao_id: str) -> bool:
        """
        Força o reprocessamento NLP de uma votação específica,
        mesmo que já tenha análise salva.
        """
        try:
            self._processar_uma(votacao_id, forcar=True)
            return True
        except Exception as e:
            logger.error(f"Falha ao reprocessar votação {votacao_id}: {e}")
            return False

    # ── Internos ──────────────────────────────────────────────────────────────

    def _processar_uma(self, votacao_id: str, forcar: bool = False):
        """Carrega dados do banco, roda NLP e salva resultado."""
        session = get_session()
        try:
            vot = session.get(Votacao, votacao_id)
            if not vot:
                logger.warning(f"Votação {votacao_id} não encontrada no banco.")
                return

            # Verifica se já foi processada (a menos que forcar=True)
            if not forcar and vot.nlp is not None:
                logger.debug(f"Votação {votacao_id} já tem NLP. Pulando.")
                return

            # Monta texto: preferência para ementa da proposição + descrição da votação
            ementa = ""
            tipo_proposicao = ""
            if vot.proposicao:
                ementa = vot.proposicao.ementa or ""
                tipo_proposicao = vot.proposicao.sigle_tipo or ""

            descricao = vot.descricao or ""

        finally:
            session.close()

        if not ementa and not descricao:
            logger.warning(f"Votação {votacao_id} sem texto para NLP.")
            return

        # Roda o pipeline NLP (modelos carregados lazy na primeira chamada)
        resultado = processar_votacao_nlp(
            ementa=ementa,
            descricao_votacao=descricao,
            resultado_aprovacao=vot.aprovacao,
            tipo_proposicao=tipo_proposicao,
        )

        # Persiste
        upsert_votacao_nlp(votacao_id, resultado)
        logger.debug(
            f"Votação {votacao_id} → tema={resultado['tema_inferido']} | "
            f"keywords={resultado['keywords'][:3]}"
        )
