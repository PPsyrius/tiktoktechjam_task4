"""Task 4 public API. Importing this package requires only the standard library."""
from .hybrid_retriever import HybridRetriever, RetrievalConfig
from .product_store import Product, ProductStore
from .types import Candidate, CandidatePool, Constraint, SearchContext, SourceHit

__all__ = ["HybridRetriever", "RetrievalConfig", "Product", "ProductStore", "Candidate", "CandidatePool",
           "Constraint", "SearchContext", "SourceHit"]
