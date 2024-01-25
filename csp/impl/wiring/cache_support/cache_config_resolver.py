from typing import List

from csp.impl.config import CacheConfig


class CacheConfigResolver:
    def __init__(self, cache_config: CacheConfig):
        from csp.impl.wiring.cache_support.graph_building import CacheCategoryOverridesTree

        self._cache_config = cache_config
        if cache_config:
            self._cache_category_overrides = CacheCategoryOverridesTree.construct_from_cache_config(cache_config)
            self._graph_overrides = getattr(cache_config, "graph_overrides", {})
        else:
            self._cache_category_overrides = None
            self._graph_overrides = None

    def resolve_cache_config(self, graph: object, category: List[str]):
        resolved_config = self._graph_overrides.get(graph, None)
        if resolved_config is None:
            resolved_config = self._cache_category_overrides.resolve_root_folder(category)
        return resolved_config
