"""Canonical catalog data layer shared by retrieval and reranking."""

from .catalog_cache import CatalogCache, CatalogCacheError
from .catalog_loader import CatalogLoader, CatalogLoadError
from .feature_extractor import FeatureExtractor
from .product import ATTRIBUTE_FIELDS, TEXT_FIELDS, Product
from .product_store import ProductStore

__all__ = [
    "ATTRIBUTE_FIELDS",
    "TEXT_FIELDS",
    "CatalogCache",
    "CatalogCacheError",
    "CatalogLoadError",
    "CatalogLoader",
    "FeatureExtractor",
    "Product",
    "ProductStore",
]
