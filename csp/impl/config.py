from typing import Dict, List

from csp.impl.managed_dataset.cache_user_custom_object_serializer import CacheObjectSerializer
from csp.impl.struct import Struct
from csp.utils.file_permissions import FilePermissions, RWXPermissions


class BaseCacheConfig(Struct):
    data_folder: str
    read_folders: List[str]  # Additional read folders from which the data should be read if available
    lock_file_permissions: FilePermissions = FilePermissions(
        user_permissions=RWXPermissions.READ | RWXPermissions.WRITE,
        group_permissions=RWXPermissions.READ | RWXPermissions.WRITE,
        others_permissions=RWXPermissions.READ | RWXPermissions.WRITE,
    )
    data_file_permissions: FilePermissions = FilePermissions(
        user_permissions=RWXPermissions.READ | RWXPermissions.WRITE,
        group_permissions=RWXPermissions.READ,
        others_permissions=RWXPermissions.READ,
    )
    merge_existing_files: bool = True


class CacheCategoryConfig(BaseCacheConfig):
    category: List[str]


class CacheConfig(BaseCacheConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "cache_serializers"):
            self.cache_serializers = {}

    allow_overwrite: bool
    # An optional override of output folders by category
    # For example:
    # category_overrides = [
    #     CacheCategoryConfig(category=['forecasts'], data_folder='possibly_group_cached_forecasts_path'),
    #     CacheCategoryConfig(category=['forecasts', 'active_research'], data_folder='possibly_user_specific_forecasts_paths'),
    # ]
    # All forecasts except for forecasts that are under active_research will be read from/written to possibly_group_cached_forecasts_path.
    # It would commonly be a path that is shared by the research team. On the other hand all forecasts under active_research will be written
    # to possibly_user_specific_forecasts_paths which can be a private path of the current user that currently researching the forecast and
    # needs to redump it often - it's not ready to share with the team yet.
    category_overrides: List[CacheCategoryConfig]
    graph_overrides: Dict[object, BaseCacheConfig]
    cache_serializers: Dict[type, CacheObjectSerializer]


class Config(Struct):
    cache_config: CacheConfig
