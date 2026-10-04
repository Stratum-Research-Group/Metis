"""
R3: HyDE + Query Decomposition for Multi-Hop Recall
====================================================
Provides query-side enhancements that improve first-hop recall:

1. **HyDE (Hypothetical Document Embeddings)**: generates a hypothetical answer 
   passage using a language model, encodes it, and uses the resulting embedding 
   for retrieval. This bridges the query-document lexical gap by placing the 
   query vector closer to relevant passages in embedding space.

2. **Query Decomposition**: decomposes multi-hop questions into atomic sub-questions 
   that can each be answered independently, then aggregates their retrieval results.

3. **Type-Routed Expansion**: routes bridge-type questions through decomposition 
   (since they have sequential reasoning chains) and comparison-type through HyDE 
   (since they need a single comprehensive passage).

Both approaches are model-agnostic and work with any SentenceTransformer encoder.
They compound with R1 (Learned Edge Scorer): better hop-1 seeds → better bridges.

Reference:
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023)
- Press et al., "Measuring and Narrowing the Compositionality Gap" (Findings of EMNLP 2023)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer


class HyDEExpander:
    """
    Hypothetical Document Embeddings (HyDE) for query expansion.
    
    Instead of using an LLM to generate hypothetical documents (which would add
    a generator dependency), this implementation uses a self-contained approach:
    
    1. **Template-based HyDE**: Uses domain-specific templates to construct 
       hypothetical answer passages from the query terms.
    2. **Multi-vector HyDE**: Generates multiple hypothetical expansions and 
       averages their embeddings for robustness.
    3. **Interpolated HyDE**: Blends the original query embedding with the 
       hypothetical document embedding using a tunable alpha parameter.
    """
    
    def __init__(
        self,
        encoder: SentenceTransformer,
        alpha: float = 0.5,
        num_expansions: int = 3,
    ):
        """
        Args:
            encoder: SentenceTransformer model for encoding passages.
            alpha: Interpolation weight between original query (0.0) and 
                   hypothetical document (1.0). Default 0.5 for equal blend.
            num_expansions: Number of hypothetical document templates to generate 
                           and average.
        """
        self.encoder = encoder
        self.alpha = alpha
        self.num_expansions = num_expansions
        
        # Domain-specific hypothesis templates
        self._templates = [
            "The answer to '{query}' is that {query_keywords}.",
            "According to research, {query_keywords} is well-established.",
            "A comprehensive study found that regarding {query}, "
            "the key finding involves {query_keywords}.",
        ]
    
    def _extract_keywords(self, query: str) -> str:
        """Extract content keywords from a query by removing question words and stopwords."""
        stopwords = {
            "what", "which", "who", "whom", "where", "when", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "has", "have",
            "had", "will", "would", "could", "should", "can", "may", "might",
            "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
            "by", "from", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "and", "but", "or",
            "so", "if", "then", "than", "that", "this", "these", "those",
            "it", "its", "be", "been", "being",
        }
        tokens = query.lower().replace("?", "").replace(".", "").split()
        keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
        return " ".join(keywords) if keywords else query
    
    def generate_hypothetical_docs(self, query: str) -> List[str]:
        """
        Generate hypothetical answer passages from the query using templates.
        
        Returns:
            List of hypothetical document strings.
        """
        keywords = self._extract_keywords(query)
        hypotheticals = []
        for i in range(min(self.num_expansions, len(self._templates))):
            template = self._templates[i]
            hypo = template.format(query=query, query_keywords=keywords)
            hypotheticals.append(hypo)
        return hypotheticals
    
    def expand_query_embedding(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute a HyDE-expanded query embedding by interpolating the original
        query vector with the average of hypothetical document vectors.
        
        Args:
            query: The original query text.
            query_embedding: Pre-computed query embedding. If None, will encode.
            
        Returns:
            L2-normalized expanded query embedding (float32).
        """
        if query_embedding is None:
            query_embedding = self.encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")[0]
        
        # Generate and encode hypothetical documents
        hypo_docs = self.generate_hypothetical_docs(query)
        hypo_embeddings = self.encoder.encode(
            hypo_docs, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        
        # Average hypothetical embeddings
        hypo_mean = np.mean(hypo_embeddings, axis=0)
        hypo_mean = hypo_mean / (np.linalg.norm(hypo_mean) + 1e-8)
        
        # Interpolate: alpha=0 → pure query, alpha=1 → pure HyDE
        expanded = (1.0 - self.alpha) * query_embedding + self.alpha * hypo_mean
        expanded = expanded / (np.linalg.norm(expanded) + 1e-8)
        
        return expanded.astype("float32")


class QueryDecomposer:
    """
    Decomposes multi-hop questions into atomic sub-questions using pattern matching.
    
    For bridge-type questions (sequential reasoning chains), the decomposer 
    identifies intermediate entities and generates sub-questions that can 
    independently retrieve relevant paragraphs.
    
    This is a lightweight, model-free approach that captures the most common 
    multi-hop patterns without requiring an LLM:
    
    Pattern types:
    1. "Who/What [verb] [entity] that [clause]?" → extract entity + clause
    2. "In what [X] did [entity] [verb]?" → entity + context
    3. Compound questions with "and" → split into parts
    """
    
    # Common bridge-type decomposition patterns
    _BRIDGE_PATTERNS = [
        # "What is the capital of the country where X was born?"
        # → ["Where was X born?", "What is the capital of [answer]?"]
        ("the country where", "the city where", "the place where",
         "the film in which", "the movie that", "the book that",
         "the team that", "the person who", "the company that"),
    ]
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose a multi-hop query into atomic sub-questions.
        
        Args:
            query: The original multi-hop question.
            
        Returns:
            List of sub-questions. If decomposition fails or is not applicable,
            returns [query] (the original question as a single-element list).
        """
        query_lower = query.lower().strip()
        
        # Strategy 1: Compound "and" split
        # "What are X and Y?" → ["What is X?", "What is Y?"]
        if " and " in query_lower and "?" in query:
            parts = self._split_compound(query)
            if len(parts) > 1:
                return parts
        
        # Strategy 2: Nested reference decomposition
        # "What is the capital of the country where Einstein was born?"
        for pattern_group in self._BRIDGE_PATTERNS:
            for pattern in pattern_group:
                if pattern in query_lower:
                    parts = self._decompose_nested(query, pattern)
                    if len(parts) > 1:
                        return parts
        
        # Strategy 3: Multi-clause with relative pronouns
        for pronoun in ["which", "whose", "where", "when"]:
            marker = f", {pronoun} "
            if marker in query_lower:
                parts = self._decompose_relative(query, marker)
                if len(parts) > 1:
                    return parts
        
        # No decomposition found — return original
        return [query]
    
    def _split_compound(self, query: str) -> List[str]:
        """Split compound questions on 'and'."""
        # Find the 'and' position
        lower = query.lower()
        and_pos = lower.find(" and ")
        if and_pos < 0:
            return [query]
        
        # Extract the question prefix (e.g., "What are")
        prefix_end = min(
            lower.find(" ") + 1 if " " in lower else len(lower),
            20,  # Limit prefix search
        )
        
        # Simple split: create two questions from the compound
        part1 = query[:and_pos].strip()
        part2 = query[and_pos + 5:].strip()
        
        # Ensure both parts are questions
        if not part1.endswith("?"):
            part1 += "?"
        if not part2.endswith("?"):
            part2 += "?"
        
        return [part1, part2]
    
    def _decompose_nested(self, query: str, pattern: str) -> List[str]:
        """Decompose nested-reference questions."""
        lower = query.lower()
        idx = lower.find(pattern)
        if idx < 0:
            return [query]
        
        # Extract the inner clause (after the pattern)
        inner = query[idx + len(pattern):].strip().rstrip("?").strip()
        
        # Extract the outer question frame (before the pattern)
        outer_frame = query[:idx].strip()
        
        # Sub-question 1: What is [inner clause]?
        sub1 = inner.capitalize()
        if not sub1.endswith("?"):
            sub1 += "?"
        
        # Sub-question 2: Original outer frame (with placeholder)
        sub2 = outer_frame + "?"
        if not sub2[0].isupper():
            sub2 = sub2[0].upper() + sub2[1:]
        
        return [sub1, sub2]
    
    def _decompose_relative(self, query: str, marker: str) -> List[str]:
        """Decompose questions with relative clause markers."""
        lower = query.lower()
        idx = lower.find(marker)
        if idx < 0:
            return [query]
        
        # Main clause: everything before the marker
        main = query[:idx].strip()
        if not main.endswith("?"):
            main += "?"
        
        # Relative clause: everything after
        relative = query[idx + len(marker):].strip()
        if not relative.endswith("?"):
            relative += "?"
        relative = relative[0].upper() + relative[1:] if relative else ""
        
        return [main, relative]


class MultiHopQueryExpander:
    """
    Combines HyDE and Query Decomposition with type-aware routing for 
    multi-hop retrieval.
    
    Routing logic:
    - **Bridge questions**: Decompose → retrieve per sub-question → merge results.
      Bridge questions have sequential reasoning chains where the answer to Q1
      provides context for Q2.
    - **Comparison questions**: HyDE expand → single retrieval with richer embedding.
      Comparison questions need a single comprehensive passage covering both entities.
    - **Unknown type**: Apply both strategies and take the union.
    
    Usage:
        expander = MultiHopQueryExpander(encoder)
        
        # For a bridge question:
        expanded_queries = expander.expand(question, q_type="bridge")
        
        # For retrieval with expansion:
        expanded_emb = expander.get_expanded_embedding(question, q_type="comparison")
    """
    
    def __init__(
        self,
        encoder: SentenceTransformer,
        hyde_alpha: float = 0.5,
        hyde_expansions: int = 3,
    ):
        self.encoder = encoder
        self.hyde = HyDEExpander(encoder, alpha=hyde_alpha, num_expansions=hyde_expansions)
        self.decomposer = QueryDecomposer()
    
    def expand(
        self,
        query: str,
        q_type: Optional[str] = None,
    ) -> List[str]:
        """
        Expand a query into sub-queries based on question type.
        
        Args:
            query: The original question.
            q_type: Question type ("bridge", "comparison", or None for auto-detect).
            
        Returns:
            List of expanded query strings. Always includes the original query.
        """
        if q_type == "bridge":
            # Decompose bridge questions
            sub_queries = self.decomposer.decompose(query)
            if len(sub_queries) > 1:
                return [query] + sub_queries  # Original + decomposed
        
        # For comparison or unknown: return original (HyDE works on embeddings)
        return [query]
    
    def get_expanded_embeddings(
        self,
        query: str,
        q_type: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get expanded query embeddings combining decomposition and HyDE.
        
        For bridge questions: returns multiple embeddings (one per sub-question),
        each HyDE-expanded.
        
        For comparison questions: returns a single HyDE-expanded embedding.
        
        Args:
            query: Original question text.
            q_type: Question type for routing.
            query_embedding: Optional pre-computed embedding of the original query.
            
        Returns:
            Array of shape (N, d_model) with N expanded embeddings.
        """
        expanded_queries = self.expand(query, q_type=q_type)
        
        # Encode all expanded queries
        all_embeddings = self.encoder.encode(
            expanded_queries, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        
        # Apply HyDE to each query embedding
        hyde_embeddings = []
        for i, eq in enumerate(expanded_queries):
            hyde_emb = self.hyde.expand_query_embedding(eq, all_embeddings[i])
            hyde_embeddings.append(hyde_emb)
        
        return np.array(hyde_embeddings, dtype="float32")
    
    def multi_query_retrieval_scores(
        self,
        query: str,
        doc_embeddings: np.ndarray,
        q_type: Optional[str] = None,
        aggregation: str = "max",
    ) -> np.ndarray:
        """
        Compute retrieval scores using expanded query embeddings.
        
        For each document, computes similarity against all expanded queries 
        and aggregates (max or mean).
        
        Args:
            query: Original question.
            doc_embeddings: (N_docs, d_model) document embeddings.
            q_type: Question type for routing.
            aggregation: "max" (take best sub-query match) or "mean" (average).
            
        Returns:
            Array of shape (N_docs,) with aggregated similarity scores.
        """
        expanded_embs = self.get_expanded_embeddings(query, q_type=q_type)
        
        # (num_queries, d_model) @ (d_model, num_docs) → (num_queries, num_docs)
        sim_matrix = np.dot(expanded_embs, doc_embeddings.T)
        
        if aggregation == "max":
            return np.max(sim_matrix, axis=0)
        elif aggregation == "mean":
            return np.mean(sim_matrix, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use 'max' or 'mean'.")
