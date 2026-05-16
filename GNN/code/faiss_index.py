"""
FAISS index manager for vulnerability detection.

Two modes:
  - Hot update (Mode A): add new embeddings without retraining the model
  - Rebuild (Mode B): rebuild the full index after LoRA fine-tuning

Stores metadata (idx, label, source code snippet) alongside each vector.
"""

import os
import json
import numpy as np
import faiss
import logging

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manages a FAISS index + metadata store for vulnerability embeddings.

    Usage:
        manager = FAISSIndexManager(embed_dim=512)
        manager.build_index(embeddings, metadata_list)
        manager.save("./faiss_store")

        manager = FAISSIndexManager.load("./faiss_store")
        results = manager.search(query_embedding, top_k=5)

        # Hot update — add new samples without retraining
        manager.add(new_embeddings, new_metadata)
        manager.save("./faiss_store")
    """

    def __init__(self, embed_dim=512, index_type="flat", nlist=100):
        """
        Args:
            embed_dim: dimensionality of embeddings
            index_type: "flat" (exact, good for <100K) or "ivf" (approximate, for >100K)
            nlist: number of Voronoi cells for IVF index
        """
        self.embed_dim = embed_dim
        self.index_type = index_type
        self.nlist = nlist
        self.metadata = []  # list of dicts: {idx, label, func_snippet}
        self.index = None
        self._build_empty_index()

    def _build_empty_index(self):
        """Create an empty FAISS index."""
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embed_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 10  # search 10 cells at query time
        else:
            # Flat inner product — exact search, good for <100K vectors
            self.index = faiss.IndexFlatIP(self.embed_dim)

    def build_index(self, embeddings, metadata_list, normalize=True):
        """
        Build/rebuild the full index from scratch.

        Args:
            embeddings: np.ndarray of shape (N, embed_dim), float32
            metadata_list: list of N dicts with keys {idx, label, func_snippet}
            normalize: L2-normalize before indexing (for cosine similarity via IP)
        """
        assert len(embeddings) == len(metadata_list), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadata"

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if normalize:
            faiss.normalize_L2(embeddings)

        # Recreate index
        self._build_empty_index()

        if self.index_type == "ivf":
            # IVF needs training
            if len(embeddings) < self.nlist:
                logger.warning(
                    f"Only {len(embeddings)} vectors but nlist={self.nlist}. "
                    f"Falling back to flat index."
                )
                self.index_type = "flat"
                self._build_empty_index()
            else:
                self.index.train(embeddings)

        self.index.add(embeddings)
        self.metadata = list(metadata_list)

        logger.info(f"Built FAISS index: {self.index.ntotal} vectors, dim={self.embed_dim}, type={self.index_type}")

    def add(self, embeddings, metadata_list, normalize=True):
        """
        Hot update — add new vectors to existing index (Mode A: zero retraining).

        Args:
            embeddings: np.ndarray (M, embed_dim)
            metadata_list: list of M metadata dicts
            normalize: L2-normalize before adding
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if normalize:
            faiss.normalize_L2(embeddings)

        if self.index_type == "ivf" and not self.index.is_trained:
            logger.warning("IVF index not trained yet. Cannot add vectors. Build first.")
            return

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

        logger.info(f"Hot update: added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5, normalize=True):
        """
        Search for nearest neighbors.

        Args:
            query_embedding: np.ndarray (1, embed_dim) or (embed_dim,)
            top_k: number of neighbors
            normalize: L2-normalize query

        Returns:
            list of dicts: [{score, idx, label, func_snippet}, ...]
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if normalize:
            faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({
                "score": float(score),
                "faiss_idx": int(idx),
                **meta
            })

        return results

    def compute_faiss_score(self, query_embedding, top_k=5, normalize=True):
        """
        Compute a FAISS-based vulnerability score.

        Returns the fraction of top-K neighbors that are labeled as vulnerable.

        Args:
            query_embedding: (embed_dim,) or (1, embed_dim)
            top_k: number of neighbors to consider

        Returns:
            float: vulnerability score in [0, 1]
        """
        if self.index.ntotal == 0:
            return 0.5  # no data — neutral

        actual_k = min(top_k, self.index.ntotal)
        results = self.search(query_embedding, top_k=actual_k, normalize=normalize)

        if len(results) == 0:
            return 0.5

        vuln_count = sum(1 for r in results if r.get("label", 0) == 1)
        return vuln_count / len(results)

    def save(self, save_dir):
        """Save index + metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)

        index_path = os.path.join(save_dir, "faiss.index")
        meta_path = os.path.join(save_dir, "metadata.json")
        config_path = os.path.join(save_dir, "config.json")

        faiss.write_index(self.index, index_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f)

        with open(config_path, "w") as f:
            json.dump({
                "embed_dim": self.embed_dim,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "ntotal": self.index.ntotal,
            }, f)

        logger.info(f"Saved FAISS index to {save_dir} ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, save_dir):
        """Load index + metadata from disk."""
        config_path = os.path.join(save_dir, "config.json")
        index_path = os.path.join(save_dir, "faiss.index")
        meta_path = os.path.join(save_dir, "metadata.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        manager = cls(
            embed_dim=config["embed_dim"],
            index_type=config["index_type"],
            nlist=config.get("nlist", 100),
        )
        manager.index = faiss.read_index(index_path)

        with open(meta_path, "r") as f:
            manager.metadata = json.load(f)

        logger.info(f"Loaded FAISS index from {save_dir} ({manager.index.ntotal} vectors)")
        return manager

    @property
    def size(self):
        return self.index.ntotal
