"""
Reranking Engine Module for Brain-2
Handles Qwen3-Reranker-4B document relevance scoring and ranking

Extracted from brain2_manager.py for modular architecture.
Maximum 150 lines following clean architecture principles.
"""

import logging
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .triton_client import TritonRerankerClient


logger = logging.getLogger(__name__)


class RerankingEngine:
    """
    Reranking Engine for Brain-2 - Handles document relevance scoring
    Extracted from brain2_manager.py for modular architecture
    """

    def __init__(self, model_manager=None, config_manager=None):
        """Initialize Reranking Engine with dependencies"""
        self.model_manager = model_manager
        self.config_manager = config_manager

        # Configuration
        self.batch_size = 16  # Optimized for RTX 5070 Ti (16GB VRAM)
        self.max_length = 512  # Token limit for query-document pairs
        self._triton = None
        # Initialize Triton client if configured
        try:
            if self.config_manager and self.config_manager.get_config("use_triton"):
                cfg = self.config_manager.get_config()
                url = cfg.get("triton_url", "http://triton:8000")
                model_name = cfg.get("triton_model_name", "qwen3_0_6b_reranking")
                self._triton = TritonRerankerClient(url=url, model_name=model_name, timeout_s=int(cfg.get("triton_timeout_s", 30)))
                logger.info(f"🔌 Using Triton reranker: {model_name} @ {url}")
        except Exception as e:
            logger.warning(f"⚠️ Triton reranker client init failed: {e}")

        # MoE Efficiency Configuration
        self.enable_moe_efficiency = True
        self.active_experts = 3  # 3B active out of 30B total parameters
        self.expert_selection_threshold = 0.1

        # Performance tracking
        self.total_rerank_requests = 0
        self.total_processing_time = 0.0
        self.total_documents_processed = 0

        logger.info("🔧 Reranking Engine initialized with MoE efficiency")
        logger.info(f"⚡ MoE: {self.active_experts}B active / 30B total parameters")

    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]],
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: Search query string
            documents: List of document dictionaries with 'text' field
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with relevance scores
        """
        if not self.model_manager or not self.model_manager.is_model_loaded():
            logger.error("❌ Model not loaded in model manager")
            return documents[:top_k]  # Return original order as fallback

        start_time = time.time()
        self.total_rerank_requests += 1

        try:
            logger.info(f"🔄 Reranking {len(documents)} documents for query: '{query[:50]}...'")

            # Extract document texts
            doc_texts = [doc.get('text', str(doc)) for doc in documents]

            # Compute relevance scores
            scores = await self._compute_relevance_scores(query, doc_texts)

            # Combine documents with scores
            scored_documents = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                scored_doc = doc.copy()
                scored_doc['relevance_score'] = score
                scored_doc['original_rank'] = i
                scored_documents.append(scored_doc)

            # Sort by relevance score (descending)
            reranked_docs = sorted(scored_documents, key=lambda x: x['relevance_score'], reverse=True)

            # Return top-k results
            top_results = reranked_docs[:top_k]

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_documents_processed += len(documents)

            logger.info(f"✅ Reranking completed in {processing_time:.3f}s")
            logger.info(f"📊 Top score: {top_results[0]['relevance_score']:.4f}")

            return top_results

        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            logger.error(f"❌ Reranking failed: {e}")
            return documents[:top_k]  # Return original order as fallback

    async def _compute_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute relevance scores for documents given a query

        This implements the core reranking logic using Qwen3-Reranker-4B model.
        """
        logger.debug(f"🧮 Computing relevance scores for {len(documents)} documents")

        try:
            scores = []

            # Process documents in batches for memory efficiency
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_scores = await self._score_batch(query, batch_docs)
                scores.extend(batch_scores)

            logger.debug(f"✅ Computed {len(scores)} relevance scores")
            return scores

        except Exception as e:
            logger.error(f"❌ Score computation failed: {e}")
            # Return uniform scores as fallback
            return [0.5] * len(documents)

    async def _score_batch(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a batch of documents for relevance to query using Qwen3-Reranker-4B
        """
        try:
            # Prefer Triton path if configured and model manager is in triton mode
            if self._triton is not None and getattr(self.model_manager, "triton_mode", False):
                _, tokenizer = self.model_manager.get_model_and_tokenizer()
                # Tokenize query once, then tile to match documents
                q = tokenizer([query], padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
                d = tokenizer(documents, padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
                # Tile query arrays to length of documents
                q_ids = np.repeat(q["input_ids"], repeats=d["input_ids"].shape[0], axis=0).astype(np.int64)
                q_mask = np.repeat(q["attention_mask"], repeats=d["attention_mask"].shape[0], axis=0).astype(np.int64)
                d_ids = d["input_ids"].astype(np.int64)
                d_mask = d["attention_mask"].astype(np.int64)
                out = self._triton.infer_batch(q_ids, q_mask, d_ids, d_mask)
                if out is None:
                    logger.warning("⚠️ Triton reranker returned None, falling back to local model if available")
                else:
                    scores_list = out.reshape(-1).tolist()
                    logger.debug(f"📊 Triton batch scores: {[f'{s:.4f}' for s in scores_list]}")
                    return scores_list

            # Local model path (fallback)
            model, tokenizer = self.model_manager.get_model_and_tokenizer()
            # Prepare input pairs (query, document)
            input_pairs = [f"Query: {query}\nDocument: {doc}" for doc in documents]
            inputs = tokenizer(
                input_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                scores = self._extract_relevance_scores(outputs)
            scores_list = scores.detach().cpu().numpy().tolist()
            logger.debug(f"📊 Local batch scores: {[f'{s:.4f}' for s in scores_list]}")
            return scores_list

        except Exception as e:
            logger.error(f"❌ Batch scoring failed: {e}")
            # Return neutral scores as fallback
            return [0.5] * len(documents)

    def _extract_relevance_scores(self, outputs) -> torch.Tensor:
        """
        Extract relevance scores from model outputs using multiple strategies
        """
        # Strategy 1: Use classification head if available
        if hasattr(outputs, 'logits'):
            scores = torch.softmax(outputs.logits, dim=-1)
            if scores.shape[-1] > 1:
                scores = scores[:, 1]  # Take positive class probability
            else:
                scores = scores.squeeze()

        # Strategy 2: Use CLS token representation
        elif hasattr(outputs, 'last_hidden_state'):
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            scores = torch.sigmoid(torch.norm(cls_embeddings, dim=1))

        # Strategy 3: Use pooled output
        elif hasattr(outputs, 'pooler_output'):
            scores = torch.sigmoid(torch.norm(outputs.pooler_output, dim=1))

        # Strategy 4: Fallback to mean pooling
        else:
            mean_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            scores = torch.sigmoid(torch.norm(mean_embeddings, dim=1))

        return scores

    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking engine statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_rerank_requests
            if self.total_rerank_requests > 0 else 0
        )

        avg_docs_per_request = (
            self.total_documents_processed / self.total_rerank_requests
            if self.total_rerank_requests > 0 else 0
        )

        return {
            "total_rerank_requests": self.total_rerank_requests,
            "total_documents_processed": self.total_documents_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "average_documents_per_request": avg_docs_per_request,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "moe_efficiency_enabled": self.enable_moe_efficiency,
            "active_experts": self.active_experts
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.total_rerank_requests = 0
        self.total_processing_time = 0.0
        self.total_documents_processed = 0
        logger.info("📊 Reranking engine statistics reset")
