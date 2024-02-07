# import collections
# import csp
# import csp.typing
# import glob
# import math
# import numpy
# import os
# import pandas
# import pytz
# import re
# import tempfile
# from typing import Dict
# import unittest
# from csp import Config, graph, node, ts
# from csp.adapters.parquet import ParquetOutputConfig
# from csp.cache_support import BaseCacheConfig, CacheCategoryConfig, CacheConfig, CacheConfigResolver, GraphCacheOptions, NoCachedDataException
# from csp.impl.managed_dataset.cache_user_custom_object_serializer import CacheObjectSerializer
# from csp.impl.managed_dataset.dataset_metadata import TimeAggregation
# from csp.impl.managed_dataset.managed_dataset_path_resolver import DatasetPartitionKey
# from csp.impl.types.instantiation_type_resolver import TSArgTypeMismatchError
# from csp.utils.object_factory_registry import Injected, register_injected_object, set_new_registry_thread_instance
# from datetime import date, datetime, timedelta
# from csp.tests.utils.typed_curve_generator import TypedCurveGenerator


# class _DummyStructWithTimestamp(csp.Struct):
#     val: int
#     timestamp: datetime


# class _GraphTempCacheFolderConfig:
#     def __init__(self, allow_overwrite=False, merge_existing_files=True):
#         self._temp_folder = None
#         self._allow_overwrite = allow_overwrite
#         self._merge_existing_files = merge_existing_files

#     def __enter__(self):
#         assert self._temp_folder is None
#         self._temp_folder = tempfile.TemporaryDirectory(prefix='csp_unit_tests')
#         return Config(cache_config=CacheConfig(data_folder=self._temp_folder.name, allow_overwrite=self._allow_overwrite,
#                                                merge_existing_files=self._merge_existing_files))

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self._temp_folder:
#             self._temp_folder.cleanup()
#             self._temp_folder = None


# @csp.node
# def csp_sorted(x: ts[['T']]) -> ts[['T']]:
#     if csp.ticked(x):
#         return sorted(x)


# class TestCaching(unittest.TestCase):

#     EXPECTED_OUTPUT_TEST_SIMPLE = {'i': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 1), (datetime(2020, 3, 1, 22, 32, 2, 2002), 2), (datetime(2020, 3, 1, 23, 33, 3, 3003), 3),
#                                          (datetime(2020, 3, 2, 0, 34, 4, 4004), 4), (datetime(2020, 3, 2, 1, 35, 5, 5005), 5), (datetime(2020, 3, 2, 2, 36, 6, 6006), 6)],
#                                    'd': [(datetime(2020, 3, 1, 21, 31, 1, 1001), date(2020, 1, 2)), (datetime(2020, 3, 1, 22, 32, 2, 2002), date(2020, 1, 3)),
#                                          (datetime(2020, 3, 1, 23, 33, 3, 3003), date(2020, 1, 4)), (datetime(2020, 3, 2, 0, 34, 4, 4004), date(2020, 1, 5)),
#                                          (datetime(2020, 3, 2, 1, 35, 5, 5005), date(2020, 1, 6)), (datetime(2020, 3, 2, 2, 36, 6, 6006), date(2020, 1, 7))],
#                                    'dt': [(datetime(2020, 3, 1, 21, 31, 1, 1001), datetime(2020, 1, 2, 0, 0, 0, 1)),
#                                           (datetime(2020, 3, 1, 22, 32, 2, 2002), datetime(2020, 1, 3, 0, 0, 0, 2)),
#                                           (datetime(2020, 3, 1, 23, 33, 3, 3003), datetime(2020, 1, 4, 0, 0, 0, 3)),
#                                           (datetime(2020, 3, 2, 0, 34, 4, 4004), datetime(2020, 1, 5, 0, 0, 0, 4)),
#                                           (datetime(2020, 3, 2, 1, 35, 5, 5005), datetime(2020, 1, 6, 0, 0, 0, 5)),
#                                           (datetime(2020, 3, 2, 2, 36, 6, 6006), datetime(2020, 1, 7, 0, 0, 0, 6))],
#                                    'f': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 0.2), (datetime(2020, 3, 1, 22, 32, 2, 2002), 0.4),
#                                          (datetime(2020, 3, 1, 23, 33, 3, 3003), 0.6000000000000001), (datetime(2020, 3, 2, 0, 34, 4, 4004), 0.8),
#                                          (datetime(2020, 3, 2, 1, 35, 5, 5005), 1.0), (datetime(2020, 3, 2, 2, 36, 6, 6006), 1.2000000000000002)],
#                                    's': [(datetime(2020, 3, 1, 21, 31, 1, 1001), '1'), (datetime(2020, 3, 1, 22, 32, 2, 2002), '2'),
#                                          (datetime(2020, 3, 1, 23, 33, 3, 3003), '3'), (datetime(2020, 3, 2, 0, 34, 4, 4004), '4'),
#                                          (datetime(2020, 3, 2, 1, 35, 5, 5005), '5'), (datetime(2020, 3, 2, 2, 36, 6, 6006), '6')],
#                                    'b': [(datetime(2020, 3, 1, 21, 31, 1, 1001), True), (datetime(2020, 3, 1, 22, 32, 2, 2002), False),
#                                          (datetime(2020, 3, 1, 23, 33, 3, 3003), True), (datetime(2020, 3, 2, 0, 34, 4, 4004), False),
#                                          (datetime(2020, 3, 2, 1, 35, 5, 5005), True), (datetime(2020, 3, 2, 2, 36, 6, 6006), False)],
#                                    'simple_leaf_node': [(datetime(2020, 3, 1, 20, 30), 1)],
#                                    'p1_i': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 33), (datetime(2020, 3, 1, 22, 32, 2, 2002), 34),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 35), (datetime(2020, 3, 2, 0, 34, 4, 4004), 36),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 37), (datetime(2020, 3, 2, 2, 36, 6, 6006), 38)],
#                                    'p1_d': [(datetime(2020, 3, 1, 21, 31, 1, 1001), date(2021, 1, 2)), (datetime(2020, 3, 1, 22, 32, 2, 2002), date(2021, 1, 3)),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), date(2021, 1, 4)), (datetime(2020, 3, 2, 0, 34, 4, 4004), date(2021, 1, 5)),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), date(2021, 1, 6)), (datetime(2020, 3, 2, 2, 36, 6, 6006), date(2021, 1, 7))],
#                                    'p1_dt': [(datetime(2020, 3, 1, 21, 31, 1, 1001), datetime(2020, 6, 7, 1, 2, 3, 5)),
#                                              (datetime(2020, 3, 1, 22, 32, 2, 2002), datetime(2020, 6, 8, 1, 2, 3, 6)),
#                                              (datetime(2020, 3, 1, 23, 33, 3, 3003), datetime(2020, 6, 9, 1, 2, 3, 7)),
#                                              (datetime(2020, 3, 2, 0, 34, 4, 4004), datetime(2020, 6, 10, 1, 2, 3, 8)),
#                                              (datetime(2020, 3, 2, 1, 35, 5, 5005), datetime(2020, 6, 11, 1, 2, 3, 9)),
#                                              (datetime(2020, 3, 2, 2, 36, 6, 6006), datetime(2020, 6, 12, 1, 2, 3, 10))],
#                                    'p1_f': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 11.4), (datetime(2020, 3, 1, 22, 32, 2, 2002), 17.1),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 22.8), (datetime(2020, 3, 2, 0, 34, 4, 4004), 28.5),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 34.2), (datetime(2020, 3, 2, 2, 36, 6, 6006), 39.900000000000006)],
#                                    'p1_s': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 'my_str1'), (datetime(2020, 3, 1, 22, 32, 2, 2002), 'my_str2'),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 'my_str3'), (datetime(2020, 3, 2, 0, 34, 4, 4004), 'my_str4'),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 'my_str5'), (datetime(2020, 3, 2, 2, 36, 6, 6006), 'my_str6')],
#                                    'p1_b': [(datetime(2020, 3, 1, 21, 31, 1, 1001), False), (datetime(2020, 3, 1, 22, 32, 2, 2002), True),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), False), (datetime(2020, 3, 2, 0, 34, 4, 4004), True),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), False), (datetime(2020, 3, 2, 2, 36, 6, 6006), True)],
#                                    'p1_simple_leaf_node': [(datetime(2020, 3, 1, 20, 30), 1)],
#                                    'p2_i': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 33), (datetime(2020, 3, 1, 22, 32, 2, 2002), 34),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 35), (datetime(2020, 3, 2, 0, 34, 4, 4004), 36),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 37), (datetime(2020, 3, 2, 2, 36, 6, 6006), 38)],
#                                    'p2_d': [(datetime(2020, 3, 1, 21, 31, 1, 1001), date(2021, 1, 3)), (datetime(2020, 3, 1, 22, 32, 2, 2002), date(2021, 1, 4)),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), date(2021, 1, 5)), (datetime(2020, 3, 2, 0, 34, 4, 4004), date(2021, 1, 6)),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), date(2021, 1, 7)), (datetime(2020, 3, 2, 2, 36, 6, 6006), date(2021, 1, 8))],
#                                    'p2_dt': [(datetime(2020, 3, 1, 21, 31, 1, 1001), datetime(2020, 6, 7, 1, 2, 3, 6)),
#                                              (datetime(2020, 3, 1, 22, 32, 2, 2002), datetime(2020, 6, 8, 1, 2, 3, 7)),
#                                              (datetime(2020, 3, 1, 23, 33, 3, 3003), datetime(2020, 6, 9, 1, 2, 3, 8)),
#                                              (datetime(2020, 3, 2, 0, 34, 4, 4004), datetime(2020, 6, 10, 1, 2, 3, 9)),
#                                              (datetime(2020, 3, 2, 1, 35, 5, 5005), datetime(2020, 6, 11, 1, 2, 3, 10)),
#                                              (datetime(2020, 3, 2, 2, 36, 6, 6006), datetime(2020, 6, 12, 1, 2, 3, 11))],
#                                    'p2_f': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 11.4), (datetime(2020, 3, 1, 22, 32, 2, 2002), 17.1),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 22.8), (datetime(2020, 3, 2, 0, 34, 4, 4004), 28.5),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 34.2), (datetime(2020, 3, 2, 2, 36, 6, 6006), 39.900000000000006)],
#                                    'p2_s': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 'my_str1'), (datetime(2020, 3, 1, 22, 32, 2, 2002), 'my_str2'),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), 'my_str3'), (datetime(2020, 3, 2, 0, 34, 4, 4004), 'my_str4'),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), 'my_str5'), (datetime(2020, 3, 2, 2, 36, 6, 6006), 'my_str6')],
#                                    'p2_b': [(datetime(2020, 3, 1, 21, 31, 1, 1001), True), (datetime(2020, 3, 1, 22, 32, 2, 2002), False),
#                                             (datetime(2020, 3, 1, 23, 33, 3, 3003), True), (datetime(2020, 3, 2, 0, 34, 4, 4004), False),
#                                             (datetime(2020, 3, 2, 1, 35, 5, 5005), True), (datetime(2020, 3, 2, 2, 36, 6, 6006), False)],
#                                    'p2_simple_leaf_node': [(datetime(2020, 3, 1, 20, 30), 1)],
#                                    'named1_i': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 2), (datetime(2020, 3, 1, 22, 32, 2, 2002), 3),
#                                                 (datetime(2020, 3, 1, 23, 33, 3, 3003), 4), (datetime(2020, 3, 2, 0, 34, 4, 4004), 5),
#                                                 (datetime(2020, 3, 2, 1, 35, 5, 5005), 6), (datetime(2020, 3, 2, 2, 36, 6, 6006), 7)],
#                                    'named1_f': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 10), (datetime(2020, 3, 1, 22, 32, 2, 2002), 20),
#                                                 (datetime(2020, 3, 1, 23, 33, 3, 3003), 30), (datetime(2020, 3, 2, 0, 34, 4, 4004), 40),
#                                                 (datetime(2020, 3, 2, 1, 35, 5, 5005), 50), (datetime(2020, 3, 2, 2, 36, 6, 6006), 60)],
#                                    'named2_i2': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 3), (datetime(2020, 3, 1, 22, 32, 2, 2002), 4),
#                                                  (datetime(2020, 3, 1, 23, 33, 3, 3003), 5), (datetime(2020, 3, 2, 0, 34, 4, 4004), 6),
#                                                  (datetime(2020, 3, 2, 1, 35, 5, 5005), 7), (datetime(2020, 3, 2, 2, 36, 6, 6006), 8)],
#                                    'named2_f2': [(datetime(2020, 3, 1, 21, 31, 1, 1001), 20), (datetime(2020, 3, 1, 22, 32, 2, 2002), 40),
#                                                  (datetime(2020, 3, 1, 23, 33, 3, 3003), 60), (datetime(2020, 3, 2, 0, 34, 4, 4004), 80),
#                                                  (datetime(2020, 3, 2, 1, 35, 5, 5005), 100), (datetime(2020, 3, 2, 2, 36, 6, 6006), 120)],
#                                    'i_sample': [(datetime(2020, 3, 1, 22, 30), 1), (datetime(2020, 3, 2, 0, 30), 3), (datetime(2020, 3, 2, 2, 30), 5)]}
#     EXPECTED_FILES = ['csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_no_part/dataset_meta.yml',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000.parquet',
#                       'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/dataset_meta.yml',
#                       'dummy_stats/sub_category/dataset1/data/2020/03/01/20200301_203000_000000-20200301_235959_999999.parquet',
#                       'dummy_stats/sub_category/dataset1/data/2020/03/02/20200302_000000_000000-20200302_030000_000000.parquet',
#                       'dummy_stats/sub_category/dataset1/dataset_meta.yml',
#                       'dummy_stats/sub_category/dataset2/data/2020/03/01/20200301_203000_000000-20200301_235959_999999.parquet',
#                       'dummy_stats/sub_category/dataset2/data/2020/03/02/20200302_000000_000000-20200302_030000_000000.parquet',
#                       'dummy_stats/sub_category/dataset2/dataset_meta.yml']
#     _SPLIT_COLUMNS_EXPECTED_FILES = ['csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_no_part/dataset_meta.yml',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/01/20200301_203000_000000-20200301_235959_999999/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210101_000000_000000/20200606_010203_000004/5.7/my_str/True/2020/03/02/20200302_000000_000000-20200302_030000_000000/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/01/20200301_203000_000000-20200301_235959_999999/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/b.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/csp_timestamp.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/d.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/dt.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/f.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/i.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/s.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/data/32/20210102_000000_000000/20200606_010203_000005/5.7/my_str/False/2020/03/02/20200302_000000_000000-20200302_030000_000000/simple_leaf_node.parquet',
#                                      'csp_unnamed_cache/test_caching.make_sub_graph_partitioned/dataset_meta.yml',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/csp_timestamp.parquet',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/f.parquet',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/i.parquet',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/csp_timestamp.parquet',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/f.parquet',
#                                      'dummy_stats/sub_category/dataset1/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/i.parquet', 'dummy_stats/sub_category/dataset1/dataset_meta.yml',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/csp_timestamp.parquet',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/f2.parquet',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/01/20200301_203000_000000-20200301_235959_999999/i2.parquet',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/csp_timestamp.parquet',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/f2.parquet',
#                                      'dummy_stats/sub_category/dataset2/data/2020/03/02/20200302_000000_000000-20200302_030000_000000/i2.parquet', 'dummy_stats/sub_category/dataset2/dataset_meta.yml']

#     class _EdgeOutputSettings(csp.Enum):
#         FIRST_CYCLE = 0x1
#         LAST_CYCLE = 0x2
#         BOTH_EDGES = FIRST_CYCLE | LAST_CYCLE

#     def _create_graph(self, split_columns_to_files):
#         func_run_count = [0]

#         def cache_options(**kwargs):
#             return GraphCacheOptions(split_columns_to_files=split_columns_to_files, **kwargs)

#         @node
#         def pass_through(v: ts['T']) -> ts['T']:
#             with csp.start():
#                 func_run_count[0] += 1

#             if csp.ticked(v):
#                 return v

#         def make_curve_pass_through(f):
#             values = [(timedelta(hours=v, minutes=v, seconds=v, milliseconds=v, microseconds=v), f(v)) for v in range(1, 20)]
#             typ = type(values[0][1])
#             return pass_through(csp.curve(typ, values))

#         simple_leaf_node = [None]

#         @graph(cache=True, cache_options=cache_options())
#         def make_sub_graph_no_part() -> csp.Outputs(i=ts[int], d=ts[date], dt=ts[datetime], f=ts[float], s=ts[str], b=ts[bool], simple_leaf_node=ts[int]):
#             return csp.output(i=make_curve_pass_through(lambda v: v),
#                        d=make_curve_pass_through(lambda v: date(2020, 1, 1) + timedelta(days=v)),
#                        dt=make_curve_pass_through(lambda v: datetime(2020, 1, 1) + timedelta(days=v, microseconds=v)),
#                        f=make_curve_pass_through(lambda v: v * .2),
#                        s=make_curve_pass_through(str),
#                        b=make_curve_pass_through(lambda v: bool(v % 2)),
#                        simple_leaf_node=simple_leaf_node[0])

#         @graph(cache=True, cache_options=cache_options())
#         def make_sub_graph_partitioned(i_v: int, d_v: date, dt_v: datetime, f_v: float, s_v: str, b_v: bool) -> csp.Outputs(i=ts[int], d=ts[date], dt=ts[datetime], f=ts[float], s=ts[str], b=ts[bool], simple_leaf_node=ts[int]):
#             no_part_sub_graph = make_sub_graph_no_part()

#             return csp.output(i=make_curve_pass_through(lambda v: i_v + v),
#                        d=make_curve_pass_through(lambda v: d_v + timedelta(days=v)),
#                        dt=make_curve_pass_through(lambda v: dt_v + timedelta(days=v, microseconds=v)),
#                        f=make_curve_pass_through(lambda v: v * f_v + f_v),
#                        s=make_curve_pass_through(lambda v: s_v + str(v)),
#                        b=make_curve_pass_through(lambda v: bool(v % 2) ^ b_v),
#                        simple_leaf_node=no_part_sub_graph.simple_leaf_node)

#         @graph(cache=True, cache_options=cache_options(dataset_name='dataset1', category=['dummy_stats', 'sub_category']))
#         def named_managed_graph_col_set_1() -> csp.Outputs(i=ts[int], f=ts[float]):
#             return csp.output(i=make_curve_pass_through(lambda v: v + 1),
#                        f=make_curve_pass_through(lambda v: v * 10.0))

#         @graph(cache=True, cache_options=cache_options(dataset_name='dataset2', category=['dummy_stats', 'sub_category']))
#         def named_managed_graph_col_set_2() -> csp.Outputs(i2=ts[int], f2=ts[float]):
#             return csp.output(i2=make_curve_pass_through(lambda v: v + 2),
#                        f2=make_curve_pass_through(lambda v: v * 20.0))

#         @graph
#         def my_graph(require_cached: bool = False):
#             self.maxDiff = 20000
#             simple_leaf_node[0] = pass_through(csp.const(1))
#             sub_graph = make_sub_graph_no_part()
#             sub_graph_partitioned = make_sub_graph_partitioned.cached if require_cached else make_sub_graph_partitioned
#             named_managed_graph_col_set_1_g = named_managed_graph_col_set_1.cached if require_cached else named_managed_graph_col_set_1
#             named_managed_graph_col_set_2_g = named_managed_graph_col_set_2.cached if require_cached else named_managed_graph_col_set_2
#             sub_graph_part_1 = sub_graph_partitioned(i_v=32, d_v=date(2021, 1, 1), dt_v=datetime(2020, 6, 6, 1, 2, 3, 4),
#                                                      f_v=5.7, s_v='my_str', b_v=True)
#             sub_graph_part_2 = sub_graph_partitioned(i_v=32, d_v=date(2021, 1, 2), dt_v=datetime(2020, 6, 6, 1, 2, 3, 5),
#                                                      f_v=5.7, s_v='my_str', b_v=False)
#             named_col_set_1 = named_managed_graph_col_set_1_g()
#             named_col_set_2 = named_managed_graph_col_set_2_g()
#             for k in sub_graph:
#                 csp.add_graph_output(k, sub_graph[k])
#             for k in sub_graph_part_1:
#                 csp.add_graph_output(f'p1_{k}', sub_graph_part_1[k])
#             for k in sub_graph_part_2:
#                 csp.add_graph_output(f'p2_{k}', sub_graph_part_2[k])
#             for k in named_col_set_1:
#                 csp.add_graph_output(f'named1_{k}', named_col_set_1[k])
#             for k in named_col_set_2:
#                 csp.add_graph_output(f'named2_{k}', named_col_set_2[k])
#             csp.add_graph_output('i_sample', pass_through(csp.sample(csp.timer(timedelta(hours=2), 1), sub_graph.i)))

#         return func_run_count, my_graph

#     def test_simple_graph(self):
#         for split_columns_to_files in (True, False):
#             with csp.memoize(False):
#                 func_run_count, my_graph = self._create_graph(split_columns_to_files=split_columns_to_files)

#                 with _GraphTempCacheFolderConfig() as config:
#                     g1 = csp.run(my_graph, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                  config=config)
#                     self.assertTrue(len(g1) > 0)
#                     func_run_count1 = func_run_count[0]
#                     # leaf node is that same in all, it's repeated 3 times
#                     self.assertEqual(len(g1) - 2, func_run_count1)
#                     self.assertEqual(g1, self.EXPECTED_OUTPUT_TEST_SIMPLE)
#                     g2 = csp.run(my_graph, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                  config=config)
#                     self.assertEqual(g1, g2)
#                     func_run_count2 = func_run_count[0]
#                     # When the sub graph is read from cache, we only have one "pass_through" for i_sample
#                     self.assertEqual(func_run_count1 + 1, func_run_count2)
#                     g3 = csp.run(my_graph, require_cached=True, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                  config=config)
#                     func_run_count3 = func_run_count[0]
#                     self.assertEqual(g1, g3)
#                     self.assertEqual(func_run_count2 + 1, func_run_count3)
#                     files_in_cache = self._get_files_in_cache(config)

#                     if split_columns_to_files:
#                         aux_files = []
#                         for f in files_in_cache:
#                             if f.endswith('.parquet'):
#                                 aux_files.append(os.path.dirname(f) + '.parquet')
#                             else:
#                                 aux_files.append(f)
#                         aux_files = sorted(set(aux_files))
#                         self.assertEqual(aux_files, self.EXPECTED_FILES)
#                         self.assertEqual(files_in_cache, self._SPLIT_COLUMNS_EXPECTED_FILES)
#                     else:
#                         self.assertEqual(files_in_cache, self.EXPECTED_FILES)

#     def _get_files_in_cache(self, config):
#         all_files_and_folders = sorted(glob.glob(f'{config.cache_config.data_folder}/**', recursive=True))
#         files_in_cache = [v.replace(f'{config.cache_config.data_folder}/', '') for v in all_files_and_folders if os.path.isfile(v)]
#         # When we right from command line, the tests import paths differ. So let's support it as well
#         files_in_cache = [f.replace('csp.tests.test_caching', 'test_caching') for f in files_in_cache]
#         files_in_cache = [f.replace('/csp.tests.', '/') for f in files_in_cache]
#         return files_in_cache

#     def test_no_cache(self):
#         for split_columns_to_files in (True, False):
#             func_run_count, my_graph_func = self._create_graph(split_columns_to_files=split_columns_to_files)
#             g1 = csp.run(my_graph_func, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390))
#             func_run_count1 = func_run_count[0]
#             g2 = csp.run(my_graph_func, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390))
#             func_run_count2 = func_run_count[0]
#             self.assertEqual(g1, g2)
#             self.assertEqual(func_run_count1 * 2, func_run_count2)

#             with self.assertRaises(NoCachedDataException):
#                 g3 = csp.run(my_graph_func, require_cached=True, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390))

#     def _get_all_files(self, config):
#         return sorted(glob.glob(f'{config.cache_config.data_folder}/**/*.parquet', recursive=True))

#     def _get_default_graph_caching_kwargs(self, split_columns_to_files):
#         if split_columns_to_files:
#             graph_kwargs = {'cache_options': GraphCacheOptions(split_columns_to_files=True)}
#         else:
#             graph_kwargs = {}
#         return graph_kwargs

#     def test_merge(self):
#         for merge_existing_files in (True, False):
#             for split_columns_to_files in (True, False):
#                 graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)

#                 def _time_to_seconds(t):
#                     return t.hour * 3600 + t.minute * 60 + t.second

#                 @csp.node()
#                 def my_node() -> csp.Outputs(hours=ts[int], minutes=ts[int], seconds=ts[int]):
#                     with csp.alarms():
#                         alarm = csp.alarm( int )
#                     with csp.start():
#                         csp.schedule_alarm(alarm, timedelta(), _time_to_seconds(csp.now()))
#                     if csp.ticked(alarm):
#                         csp.schedule_alarm(alarm, timedelta(seconds=60), alarm + 60)
#                         return csp.output(hours=alarm // 3600, minutes=alarm // 60, seconds=alarm)

#                 @csp.graph(cache=True, **graph_kwargs)
#                 def sub_graph() -> csp.Outputs(hours=ts[int], minutes=ts[int], seconds=ts[int]):
#                     node = my_node()
#                     return csp.output(hours=node.hours, minutes=node.minutes, seconds=node.seconds)

#                 def _validate_file_df(g, start_time, dt, g_start=None, g_end=None):
#                     end_time = start_time + dt if isinstance(dt, timedelta) else dt
#                     g_start = g_start if g_start else start_time
#                     g_end = g_end if g_end else end_time
#                     g_end = g_start + g_end if isinstance(g_end, timedelta) else g_end
#                     df = sub_graph.cached_data(config.cache_config.data_folder)().get_data_df_for_period(start_time, dt)

#                     self.assertTrue((df.seconds.diff()[1:] == 60).all())
#                     self.assertTrue((df.minutes == df.seconds // 60).all())
#                     self.assertTrue((df.hours == df.seconds // 3600).all())
#                     self.assertTrue(df.iloc[-1]['csp_timestamp'] == end_time)
#                     self.assertTrue(df.iloc[0]['csp_timestamp'] == start_time)
#                     self.assertTrue(df.iloc[0]['seconds'] == _time_to_seconds(start_time))
#                     self.assertEqual(g['seconds'][0][1], _time_to_seconds(g_start))
#                     self.assertEqual(g['seconds'][-1][1], _time_to_seconds(g_end))

#                 def graph():
#                     res = sub_graph()
#                     csp.add_graph_output('seconds', res.seconds)

#                 with _GraphTempCacheFolderConfig(allow_overwrite=True, merge_existing_files=merge_existing_files) as config:
#                     missing_range_handler = lambda start, end: True
#                     start_time1 = datetime(2020, 3, 1, 9, 30, tzinfo=pytz.utc)
#                     dt1 = timedelta(hours=0, minutes=60)
#                     g = csp.run(graph, starttime=start_time1, endtime=dt1, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     self.assertEqual(len(files), 1)
#                     _validate_file_df(g, start_time1, dt1)

#                     start_time2 = start_time1 + timedelta(minutes=180)
#                     dt2 = dt1
#                     g = csp.run(graph, starttime=start_time2, endtime=dt2, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())

#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 2)
#                     _validate_file_df(g, start_time2, dt2)
#                     # Test repeated writing of the same file
#                     g = csp.run(graph, starttime=start_time2, endtime=dt2, config=config)
#                     _validate_file_df(g, start_time2, dt2)

#                     start_time3 = start_time2 + dt2 - timedelta(minutes=5)
#                     dt3 = timedelta(minutes=15)
#                     g = csp.run(graph, starttime=start_time3, endtime=dt3, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())

#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 2)
#                     _validate_file_df(g, start_time2, start_time3 + dt3, g_start=start_time3, g_end=dt3)

#                     start_time4 = start_time2 - timedelta(minutes=5)
#                     dt4 = timedelta(minutes=15)
#                     g = csp.run(graph, starttime=start_time4, endtime=dt4, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 2)
#                     _validate_file_df(g, start_time4, start_time3 + dt3, g_start=start_time4, g_end=dt4)

#                     start_time5 = start_time1 + timedelta(minutes=40)
#                     dt5 = timedelta(minutes=200)
#                     g = csp.run(graph, starttime=start_time5, endtime=dt5, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 1)
#                     _validate_file_df(g, start_time1, start_time3 + dt3, g_start=start_time5, g_end=dt5)

#                     g = csp.run(graph, starttime=start_time1 + timedelta(minutes=10), endtime=dt1, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 1)
#                     _validate_file_df(g, start_time1, start_time3 + dt3, g_start=start_time1 + timedelta(minutes=10), g_end=dt1)

#                     start_time6 = start_time1 - timedelta(minutes=10)
#                     dt6 = start_time3 + dt3 + timedelta(minutes=10)
#                     g = csp.run(graph, starttime=start_time6, endtime=dt6, config=config)
#                     files = list(sub_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     self.assertEqual(len(files), 1)
#                     _validate_file_df(g, start_time6, dt6)

#     def test_folder_overrides(self):
#         for split_columns_to_files in (True, False):
#             start_time = datetime(2020, 3, 1, 20, 30)
#             end_time = start_time + timedelta(seconds=1)

#             @csp.graph(cache=True)
#             def g1() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(category=['C1', 'C2', 'C3'], split_columns_to_files=split_columns_to_files))
#             def g2() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(category=['C1', 'C2', 'C3_2'], split_columns_to_files=split_columns_to_files))
#             def g3() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(category=['C1', 'C2'], split_columns_to_files=split_columns_to_files))
#             def g4() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(category=['C1'], split_columns_to_files=split_columns_to_files))
#             def g5() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(dataset_name='named_dataset1', split_columns_to_files=split_columns_to_files))
#             def g6() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(dataset_name='named_dataset2', category=['C1', 'C2'], split_columns_to_files=split_columns_to_files))
#             def g7() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph(cache=True, cache_options=GraphCacheOptions(dataset_name='named_dataset3', category=['C1', 'C2'], split_columns_to_files=split_columns_to_files))
#             def g8() -> csp.Outputs(o=csp.ts[int]):
#                 return csp.output(o=csp.null_ts(int))

#             @csp.graph
#             def g():
#                 g1(), g2(), g3(), g4(), g5(), g6(), g7(), g8()

#             def _get_data_folders_for_config(config):
#                 all_folders = sorted({os.path.dirname(v) for v in self._get_files_in_cache(config)})
#                 return sorted({v[:v.index('/data')] for v in all_folders if '/data' in v})

#             with _GraphTempCacheFolderConfig() as config:
#                 with _GraphTempCacheFolderConfig() as config2:
#                     config_copy = Config.from_dict(config.to_dict())
#                     root_folder = config_copy.cache_config.data_folder
#                     config_copy.cache_config.data_folder = os.path.join(root_folder, "default_output_folder")
#                     config_copy.cache_config.category_overrides = [
#                         CacheCategoryConfig(category=['C1'],
#                                             data_folder=os.path.join(root_folder, 'C1_O')),
#                         CacheCategoryConfig(category=['C1', 'C2'],
#                                             data_folder=os.path.join(root_folder, 'C1_C2_O')),
#                         CacheCategoryConfig(category=['C1', 'C2', 'C3'],
#                                             data_folder=os.path.join(root_folder, 'C1_C2_C3_O'))
#                     ]
#                     config_copy.cache_config.graph_overrides = {g8: BaseCacheConfig(data_folder=config2.cache_config.data_folder)}
#                     csp.run(g, starttime=start_time, endtime=end_time, config=config_copy)
#                     data_folders = _get_data_folders_for_config(config)
#                     data_folders2 = _get_data_folders_for_config(config2)
#                     expected_dataset_folders = {
#                         'g1': 'default_output_folder/csp_unnamed_cache/test_caching.g1', 'g2': 'C1_C2_C3_O/C1/C2/C3/test_caching.g2',
#                         'g3': 'C1_C2_O/C1/C2/C3_2/test_caching.g3', 'g4': 'C1_C2_O/C1/C2/test_caching.g4', 'g5': 'C1_O/C1/test_caching.g5',
#                         'g6': 'default_output_folder/csp_unnamed_cache/named_dataset1', 'g7': 'C1_C2_O/C1/C2/named_dataset2'}
#                     expected_dataset_folders2 = {'g8': 'C1/C2/named_dataset3'}
#                     self.assertEqual(data_folders, sorted(expected_dataset_folders.values()))
#                     self.assertEqual(data_folders2, sorted(expected_dataset_folders2.values()))

#                     full_path = lambda v: os.path.join(root_folder, v)
#                     get_data_files = lambda g, f: g.cached_data(full_path(f))().get_data_files_for_period(start_time, end_time)

#                     self.assertEqual(1, len(get_data_files(g1, "default_output_folder")))
#                     self.assertEqual(1, len(get_data_files(g2, "C1_C2_C3_O")))
#                     self.assertEqual(1, len(get_data_files(g3, "C1_C2_O")))
#                     self.assertEqual(1, len(get_data_files(g4, "C1_C2_O")))
#                     self.assertEqual(1, len(get_data_files(g5, "C1_O")))
#                     self.assertEqual(1, len(get_data_files(g6, "default_output_folder")))
#                     self.assertEqual(1, len(get_data_files(g7, "C1_C2_O")))

#                     data_path_resolver = CacheConfigResolver(config_copy.cache_config)
#                     get_data_files = lambda g: g.cached_data(data_path_resolver)().get_data_files_for_period(start_time, end_time)
#                     self.assertEqual(1, len(get_data_files(g1)))
#                     self.assertEqual(1, len(get_data_files(g2)))
#                     self.assertEqual(1, len(get_data_files(g3)))
#                     self.assertEqual(1, len(get_data_files(g4)))
#                     self.assertEqual(1, len(get_data_files(g5)))
#                     self.assertEqual(1, len(get_data_files(g6)))
#                     self.assertEqual(1, len(get_data_files(g7)))
#                     self.assertEqual(1, len(get_data_files(g8)))

#     def test_caching_reads_only_needed_columns(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)

#             class MyS(csp.Struct):
#                 x: int
#                 y: int

#             @graph(cache=True, **graph_kwargs)
#             def g(s: str) -> csp.Outputs(o=csp.ts[MyS]):
#                 t = csp.engine_start_time()
#                 o_ts = csp.curve(MyS, [(t + timedelta(seconds=v), MyS(x=v, y=v * 2)) for v in range(20)])
#                 return csp.output(o=o_ts)

#             @graph
#             def g_x_reader(s: str) -> csp.Outputs(o=csp.ts[int]):
#                 return csp.count(g('A').o.x)

#             @graph
#             def g_delayed_demux(s: str) -> csp.ts[int]:
#                 demux = csp.DelayedDemultiplex(g('A').o.x, g('A').o.x)
#                 return csp.count(demux.demultiplex(1))

#             with _GraphTempCacheFolderConfig() as config:
#                 csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 files = g.cached_data(config.cache_config.data_folder)('A').get_data_files_for_period(datetime(2020, 1, 1), datetime(2020, 1, 1) + timedelta(seconds=20))
#                 self.assertEqual(len(files), 1)
#                 file = next(iter(files.values()))
#                 if split_columns_to_files:
#                     # Let's fake the data file by removing the column y
#                     file_to_remove = os.path.join(file, 'o.y.parquet')
#                     self.assertTrue(os.path.exists(file_to_remove))
#                     os.unlink(file_to_remove)
#                     self.assertFalse(os.path.exists(file_to_remove))
#                     with self.assertRaisesRegex(Exception, r'.*IOError.*Failed to open .*o\.y.*'):
#                         csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 else:
#                     df = pandas.read_parquet(file)
#                     # Let's fake the data file by removing the column y. We want to make sure that we don't attempt to read column y
#                     df.drop(columns=['o.y']).to_parquet(file)
#                     with self.assertRaisesRegex(RuntimeError, r'Missing column o\.y.*'):
#                         csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 # This should not raise since we don't try to read the y column
#                 csp.run(g_x_reader, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 csp.run(g_delayed_demux, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)

#     def test_enum_serialization(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)

#             class MyEnum(csp.Enum):
#                 X = csp.Enum.auto()
#                 Y = csp.Enum.auto()
#                 ZZZ = csp.Enum.auto()

#             raiseExc = [False]

#             @graph(cache=True, **graph_kwargs)
#             def g(s: str) -> csp.Outputs(o=csp.ts[MyEnum]):
#                 if raiseExc[0]:
#                     raise RuntimeError("Shouldn't get here")
#                 o_ts = csp.curve(MyEnum, [(timedelta(seconds=1), MyEnum.X), (timedelta(seconds=1), MyEnum.Y), (timedelta(seconds=2), MyEnum.ZZZ), (timedelta(seconds=3), MyEnum.X)])
#                 return csp.output(o=o_ts)

#             from csp.utils.qualified_name_utils import QualifiedNameUtils
#             QualifiedNameUtils.register_type(MyEnum)

#             with _GraphTempCacheFolderConfig() as config:
#                 csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 raiseExc[0] = True
#                 cached_res = csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 enum_values = [v[1] for v in cached_res['o']]
#                 data_df = g.cached_data(config.cache_config.data_folder)('A').get_data_df_for_period()
#                 self.assertEqual(data_df['o'].tolist(), ['X', 'Y', 'ZZZ', 'X'])
#                 self.assertEqual(enum_values, [MyEnum.X, MyEnum.Y, MyEnum.ZZZ, MyEnum.X])

#     def test_enum_field_serialization(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)
#             from csp.tests.impl.test_enum import MyEnum

#             class MyStruct(csp.Struct):
#                 e: MyEnum

#             raiseExc = [False]

#             @graph(cache=True, **graph_kwargs)
#             def g(s: str) -> csp.Outputs(o=csp.ts[MyStruct]):
#                 if raiseExc[0]:
#                     raise RuntimeError("Shouldn't get here")
#                 make_s = lambda v: MyStruct(e=v) if v is not None else MyStruct()
#                 o_ts = csp.curve(MyStruct, [(timedelta(seconds=1), make_s(MyEnum.A)), (timedelta(seconds=1), make_s(MyEnum.B)),
#                                             (timedelta(seconds=2), make_s(MyEnum.C)), (timedelta(seconds=3), make_s(MyEnum.A)),
#                                             (timedelta(seconds=4), make_s(None))])
#                 return csp.output(o=o_ts)

#             with _GraphTempCacheFolderConfig() as config:
#                 csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 raiseExc[0] = True
#                 cached_res = csp.run(g, 'A', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 enum_values = [v[1].e if hasattr(v[1], 'e') else None for v in cached_res['o']]
#                 data_df = g.cached_data(config.cache_config.data_folder)('A').get_data_df_for_period()
#                 self.assertEqual(data_df['o.e'].tolist(), ['A', 'B', 'C', 'A', None])
#                 self.assertEqual(enum_values, [MyEnum.A, MyEnum.B, MyEnum.C, MyEnum.A, None])

#     def test_nested_struct_caching(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)
#             from csp.tests.impl.test_enum import MyEnum
#             class MyStruct1(csp.Struct):
#                 v_int: int
#                 v_str: str
#                 e: MyEnum

#             class MyStruct2(csp.Struct):
#                 v: MyStruct1
#                 v_float: float

#             class MyStruct3(MyStruct2):
#                 v2: MyStruct2

#             from csp.utils.qualified_name_utils import QualifiedNameUtils
#             QualifiedNameUtils.register_type(MyStruct1)
#             QualifiedNameUtils.register_type(MyStruct2)

#             raiseExc = [False]

#             struct_values = [MyStruct3(),
#                              MyStruct3(v=MyStruct1(v_int=1)),
#                              MyStruct3(v=MyStruct1(v_int=2)),
#                              MyStruct3(v=MyStruct1(v_int=3, v_str='3_val')),
#                              MyStruct3(v=MyStruct1(v_str='4_val')),
#                              MyStruct3(v=MyStruct1(v_str='5_val'), v2=MyStruct2(v_float=5.5, v=MyStruct1(v_int=6, v_str='6_val', e=MyEnum.B)), v_float=6.5),
#                              MyStruct3(v=MyStruct1())
#                              ]

#             @graph(cache=True, **graph_kwargs)
#             def g() -> csp.Outputs(o=csp.ts[MyStruct3]):
#                 if raiseExc[0]:
#                     raise RuntimeError("Shouldn't get here")
#                 o_ts = csp.curve(MyStruct3, [(timedelta(seconds=i), v) for i, v in enumerate(struct_values)])
#                 return csp.output(o=o_ts)

#             @graph
#             def g2():
#                 csp.add_graph_output('o', g().o)
#                 csp.add_graph_output('o.v', g().o.v)
#                 csp.add_graph_output('o.v_float', g().o.v_float)

#             @graph
#             def g3():
#                 csp.add_graph_output('o.v_float', g().o.v_float)

#             with _GraphTempCacheFolderConfig() as config:
#                 csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 raiseExc[0] = True
#                 cached_res = csp.run(g2, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 cached_float = csp.run(g3, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 cached_values = list(zip(*cached_res['o']))[1]
#                 cached_v_values = list(zip(*cached_res['o.v']))[1]
#                 expected_v_values = [getattr(v, 'v') for v in cached_values if hasattr(v, 'v')]
#                 self.assertEqual(len(struct_values), len(cached_values))
#                 for v1, v2 in zip(struct_values, cached_values):
#                     self.assertEqual(v1, v2)
#                 self.assertEqual(len(cached_v_values), len(expected_v_values))
#                 for v1, v2 in zip(cached_v_values, expected_v_values):
#                     self.assertEqual(v1, v2)
#                 self.assertEqual(cached_float['o.v_float'], cached_res['o.v_float'])

#     def test_caching_same_timestamp_with_missing_values(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)

#             @csp.node
#             def my_node() -> csp.Outputs(v1=csp.ts[int], v2=csp.ts[int], v3=csp.ts[int]):
#                 with csp.alarms():
#                     a = csp.alarm( int )
#                 with csp.start():
#                     csp.schedule_alarm(a, timedelta(0), 0)
#                     csp.schedule_alarm(a, timedelta(0), 1)
#                     csp.schedule_alarm(a, timedelta(0), 2)
#                     csp.schedule_alarm(a, timedelta(0), 3)
#                 if csp.ticked(a):
#                     if a == 0:
#                         csp.output(v1=10 + a, v2=20 + a)
#                     elif a == 1:
#                         csp.output(v1=10 + a, v3=30 + a)
#                     else:
#                         csp.output(v1=10 + a, v2=20 + a, v3=30 + a)

#             @graph(cache=True, **graph_kwargs)
#             def g() -> csp.Outputs(v1=csp.ts[int], v2=csp.ts[int], v3=csp.ts[int]):
#                 outs = my_node()
#                 return csp.output(v1=outs.v1, v2=outs.v2, v3=outs.v3)

#             @graph
#             def main():
#                 csp.add_graph_output('l', csp_sorted(csp.collect([g().v1, g().v2, g().v3])))

#             with _GraphTempCacheFolderConfig() as config:
#                 out1 = csp.run(main, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 out2 = csp.run(main, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)
#                 self.assertEqual(out1, out2)

#     def test_timestamp_with_nanos_caching(self):
#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)
#             timestamp_value = pandas.Timestamp('2020-01-01 00:00:00') + pandas.to_timedelta(123, 'ns')

#             @csp.node
#             def my_node() -> csp.ts[datetime]:
#                 with csp.alarms():
#                     a = csp.alarm( datetime )
#                 with csp.start():
#                     csp.schedule_alarm(a, timedelta(seconds=1), timestamp_value)
#                 if csp.ticked(a):
#                     return a

#             @graph(cache=True, **graph_kwargs)
#             def g() -> csp.Outputs(t=csp.ts[datetime]):
#                 return csp.output(t=my_node())

#             with _GraphTempCacheFolderConfig() as config:
#                 csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20), config=config)

#                 data_path_resolver = CacheConfigResolver(config.cache_config)

#                 data_df = g.cached_data(data_path_resolver)().get_data_df_for_period()

#                 self.assertEqual(timestamp_value.nanosecond, 123)
#                 self.assertEqual(len(data_df), 1)
#                 self.assertEqual(data_df.t.iloc[0].tz_localize(None), timestamp_value)

#     def test_unsupported_basket_caching(self):
#         with self.assertRaisesRegex(NotImplementedError, "Caching of list basket outputs is unsupported"):
#             @csp.graph(cache=True)
#             def g_bad() -> csp.Outputs(list_basket=[csp.ts[str]]):
#                 raise RuntimeError()

#         with self.assertRaisesRegex(TypeError, "Cached output basket dict_basket must have shape provided using with_shape or with_shape_of"):
#             @csp.graph(cache=True)
#             def g_bad() -> csp.Outputs(dict_basket=csp.OutputBasket(Dict[str, csp.ts[str]])):
#                 raise RuntimeError()

#         with self.assertRaisesRegex(RuntimeError, "Cached graph with output basket must set split_columns_to_files to True"):
#             @csp.graph(cache=True, cache_options=GraphCacheOptions(split_columns_to_files=False))
#             def g_bad() -> csp.Outputs(dict_basket=csp.OutputBasket(Dict[str, csp.ts[str]], shape=[1,2,3])):
#                 raise RuntimeError()
#         # TODO: add shape validation check here

#     def test_simple_dict_basket_caching(self):
#         def shape_func(l=None):
#             if l is None:
#                 return ['x', 'y', 'z']
#             return l

#         @csp.node
#         def my_node() -> csp.Outputs(scalar1=csp.ts[int], dict_basket=
#                                     csp.OutputBasket(Dict[str, csp.ts[int]], shape=shape_func()), scalar=csp.ts[int]):
#             with csp.alarms():
#                 a_index = csp.alarm( int )
#             with csp.start():
#                 csp.schedule_alarm(a_index, timedelta(), 0)
#             if csp.ticked(a_index) and a_index < 10:
#                 if a_index == 1:
#                     csp.schedule_alarm(a_index, timedelta(), 2)
#                 else:
#                     csp.schedule_alarm(a_index, timedelta(seconds=1), a_index + 1)

#             if a_index == 0:
#                 csp.output(scalar1=1, dict_basket={'x': 1, 'y': 2, 'z': 3}, scalar=2)
#             elif a_index == 1:
#                 csp.output(dict_basket={'x': 2, 'z': 3}, scalar=3)
#             elif a_index == 2:
#                 csp.output(dict_basket={'x': 3, 'z': 34})
#             elif a_index == 3:
#                 csp.output(scalar1=5)
#             elif a_index == 4:
#                 csp.output(dict_basket={'x': 45})

#         @csp.graph(cache=True)
#         def g_bad() -> csp.Outputs(scalar1=csp.ts[int], dict_basket=csp.OutputBasket(Dict[str, csp.ts[int]], shape=shape_func()), scalar=csp.ts[int]):
#             # __outputs__(dict_basket={'T': csp.ts['K']}.with_shape(shape_func(['xx'])))
#             #
#             # return csp.output( dict_basket={'xx': csp.const(1)})

#             return csp.output(scalar1=my_node().scalar1, dict_basket=my_node().dict_basket, scalar=my_node().scalar)

#         # @csp.node
#         # def g_bad():
#         #     __outputs__(scalar1=csp.ts[int], dict_basket={'T': csp.ts['K']}.with_shape(shape_func()), scalar=csp.ts[int])
#         #     return csp.output(scalar1=5, dict_basket={'x': 1},  scalar=3)

#         @graph
#         def run_graph(g: object):
#             g_bad()

#         with _GraphTempCacheFolderConfig() as config:
#             csp.run(run_graph, g_bad, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)

#     def test_simple_basket_caching(self):
#         for typ in (int, bool, float, str, datetime, date, TypedCurveGenerator.SimpleEnum, TypedCurveGenerator.SimpleStruct, TypedCurveGenerator.NestedStruct):
#             @graph(cache=True)
#             def cached_graph() -> csp.Outputs(v1=csp.OutputBasket(Dict[str, csp.ts[typ]], shape=['0', '', '2']), v2=csp.ts[int]):
#                 curve_generator = TypedCurveGenerator()

#                 return csp.output(v1={'0': curve_generator.gen_transformed_curve(typ, 0, 100, 1, skip_indices=[5, 6, 7], duplicate_timestamp_indices=[8, 9]),
#                                '': curve_generator.gen_transformed_curve(typ, 13, 100, 1, skip_indices=[5, 7, 9]),
#                                '2': curve_generator.gen_transformed_curve(typ, 27, 100, 1, skip_indices=[5, 6])
#                                },
#                            v2=curve_generator.gen_int_curve(100, 10, 1, skip_indices=[2], duplicate_timestamp_indices=[7, 8]))

#             @graph
#             def run_graph(force_cached: bool = False):
#                 g = cached_graph.cached if force_cached else cached_graph
#                 csp.add_graph_output('v1[0]', g().v1['0'])
#                 csp.add_graph_output('v1[1]', g().v1[''])
#                 csp.add_graph_output('v1[2]', g().v1['2'])
#                 csp.add_graph_output('v2', g().v2)

#             with _GraphTempCacheFolderConfig() as config:
#                 res = csp.run(run_graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                 res2 = csp.run(run_graph, True, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                 self.assertEqual(res, res2)

#     def test_basket_caching_first_last_cycles(self):
#         for basket_edge_settings in self._EdgeOutputSettings:
#             for scalar_edge_settings in self._EdgeOutputSettings:
#                 @graph(cache=True)
#                 def cached_graph(basket_edge_settings: self._EdgeOutputSettings, scalar_edge_settings: self._EdgeOutputSettings) -> csp.Outputs(
#                         v1=csp.OutputBasket(Dict[str, csp.ts[int]], shape=['0', '1']), v2=csp.ts[int]):
#                     curve_generator = TypedCurveGenerator()
#                     output_scalar_on_initial_cycle = bool(scalar_edge_settings.value & self._EdgeOutputSettings.FIRST_CYCLE.value)
#                     output_basket_on_initial_cycle = bool(basket_edge_settings.value & self._EdgeOutputSettings.FIRST_CYCLE.value)
#                     basket_skip_indices = [] if bool(basket_edge_settings.value & self._EdgeOutputSettings.LAST_CYCLE.value) else [3]
#                     scalar_skip_indices = [] if bool(scalar_edge_settings.value & self._EdgeOutputSettings.LAST_CYCLE.value) else [3]

#                     return csp.output(v1={'0': curve_generator.gen_int_curve(0, 3, 1, output_on_initial_cycle=output_basket_on_initial_cycle, skip_indices=basket_skip_indices),
#                                    '1': curve_generator.gen_int_curve(13, 3, 1, output_on_initial_cycle=output_basket_on_initial_cycle, skip_indices=basket_skip_indices)},
#                                v2=curve_generator.gen_int_curve(100, 3, 1, output_on_initial_cycle=output_scalar_on_initial_cycle, skip_indices=scalar_skip_indices))

#                 @graph
#                 def run_graph(basket_edge_settings: self._EdgeOutputSettings, scalar_edge_settings: self._EdgeOutputSettings, force_cached: bool = False):
#                     g = cached_graph.cached if force_cached else cached_graph
#                     g_res = g(basket_edge_settings, scalar_edge_settings)
#                     csp.add_graph_output('v1[0]', g_res.v1['0'])
#                     csp.add_graph_output('v1[1]', g_res.v1['1'])
#                     csp.add_graph_output('v2', g_res.v2)

#                 with _GraphTempCacheFolderConfig() as config:
#                     res = csp.run(run_graph, basket_edge_settings, scalar_edge_settings, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                     res2 = csp.run(run_graph, basket_edge_settings, scalar_edge_settings, True, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                     self.assertEqual(res, res2)

#     def test_basket_multiday_read_write(self):
#         # 5 hours - means we will have data each day and the last few days are empty. With 49 hours we will have some days in the middle having empty
#         # data and we want to see that it's handled properly (this test actually found a hidden bug)
#         for curve_hours in (5, 49):
#             for typ in (int, bool, float, str, datetime, date, TypedCurveGenerator.SimpleEnum, TypedCurveGenerator.SimpleStruct, TypedCurveGenerator.NestedStruct):
#                 @graph(cache=True)
#                 def cached_graph() ->  csp.Outputs(v1=csp.OutputBasket(Dict[str, csp.ts[typ]], shape=['0', '', '2']), v2=csp.ts[int]):
#                     curve_generator = TypedCurveGenerator(period=timedelta(hours=curve_hours))

#                     return csp.output(v1={'0': curve_generator.gen_transformed_curve(typ, 0, 10, 1, skip_indices=[5, 6, 7], duplicate_timestamp_indices=[8, 9]),
#                                    '': curve_generator.gen_transformed_curve(typ, 13, 10, 1, skip_indices=[5, 7, 9]),
#                                    '2': curve_generator.gen_transformed_curve(typ, 27, 10, 1, skip_indices=[5, 6])
#                                    },
#                                v2=curve_generator.gen_int_curve(100, 10, 1, skip_indices=[2], duplicate_timestamp_indices=[7, 8]))

#                 @graph
#                 def run_graph(force_cached: bool = False):
#                     g = cached_graph.cached if force_cached else cached_graph
#                     csp.add_graph_output('v1[0]', g().v1['0'])
#                     csp.add_graph_output('v1[1]', g().v1[''])
#                     csp.add_graph_output('v1[2]', g().v1['2'])
#                     csp.add_graph_output('v2', g().v2)

#                 self.maxDiff = None
#                 with _GraphTempCacheFolderConfig() as config:
#                     res = csp.run(run_graph, starttime=datetime(2020, 1, 1), endtime=timedelta(days=5) - timedelta(microseconds=1), config=config)
#                     res2 = csp.run(run_graph, True, starttime=datetime(2020, 1, 1), endtime=timedelta(days=5) - timedelta(microseconds=1), config=config)
#                     self.assertEqual(res, res2)
#                     data_path_resolver = CacheConfigResolver(config.cache_config)
#                     # A sanity check that we can load the data with some empty dataframes on some days
#                     base_data_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period()

#     def test_merge_baskets(self):
#         def _simple_struct_to_dict(o):
#             if o is None:
#                 return None
#             return {c: getattr(o, c, None) for c in TypedCurveGenerator.SimpleStruct.metadata()}

#         for batch_size in (117, None):
#             output_config = ParquetOutputConfig() if batch_size is None else ParquetOutputConfig(batch_size=batch_size)
#             for typ in (int, bool, float, str, datetime, date, TypedCurveGenerator.SimpleEnum, TypedCurveGenerator.SimpleStruct, TypedCurveGenerator.NestedStruct):
#                 @graph(cache=True)
#                 def base_graph() -> csp.Outputs(v1=csp.OutputBasket(Dict[str, csp.ts[typ]], shape=['COL1', 'COL2', 'COL3']), v2=csp.ts[int]):
#                     curve_generator = TypedCurveGenerator(period=timedelta(seconds=7))
#                     return csp.output(v1={'COL1': curve_generator.gen_transformed_curve(typ, 0, 2600, 1, skip_indices=[95, 96, 97], duplicate_timestamp_indices=[98, 99]),
#                                    'COL2': curve_generator.gen_transformed_curve(typ, 13, 2600, 1, skip_indices=[95, 97, 99, 1090]),
#                                    'COL3': curve_generator.gen_transformed_curve(typ, 27, 2600, 1, skip_indices=[95, 96])
#                                    },
#                                v2=curve_generator.gen_int_curve(100, 2600, 1, skip_indices=[92], duplicate_timestamp_indices=[97, 98]))

#                 @graph(cache=True, cache_options=GraphCacheOptions(parquet_output_config=output_config))
#                 def cached_graph() -> csp.Outputs(v1=csp.OutputBasket(Dict[str, csp.ts[typ]], shape=['COL1', 'COL2', 'COL3']), v2=csp.ts[int]):
#                     return csp.output(v1=base_graph.cached().v1,
#                                v2=base_graph.cached().v2)

#                 @graph
#                 def run_graph(force_cached: bool = False):
#                     g = cached_graph.cached if force_cached else cached_graph
#                     csp.add_graph_output('COL1', g().v1['COL1'])
#                     csp.add_graph_output('COL2', g().v1['COL2'])
#                     csp.add_graph_output('COL3', g().v1['COL3'])
#                     csp.add_graph_output('v2', g().v2)

#                 # enough to check this just for one type
#                 merge_existing_files = typ is int
#                 with _GraphTempCacheFolderConfig(allow_overwrite=True, merge_existing_files=merge_existing_files) as config:
#                     base_data_outputs = csp.run(base_graph, starttime=datetime(2020, 3, 1, 9, 20, tzinfo=pytz.utc),
#                                                 endtime=datetime(2020, 3, 1, 14, 0, tzinfo=pytz.utc),
#                                                 config=config)

#                     aux_dfs = [pandas.DataFrame(dict(zip(['csp_timestamp', k], zip(*v)))) for k, v in base_data_outputs.items()]
#                     for aux_df in aux_dfs:
#                         repeated_timestamp_mask = 1 - (aux_df['csp_timestamp'].shift(1) != aux_df['csp_timestamp']).astype(int)
#                         aux_df['cycle_count'] = repeated_timestamp_mask.cumsum() * repeated_timestamp_mask
#                         aux_df.set_index(['csp_timestamp', 'cycle_count'], inplace=True)

#                     # this does not work as of pandas==1.4.0
#                     # expected_base_df = pandas.concat(aux_dfs, axis=1)
#                     expected_base_df = aux_dfs[0]
#                     for df in aux_dfs[1:]:
#                         expected_base_df = expected_base_df.merge(df, left_index=True, right_index=True, how="outer")
#                     expected_base_df = expected_base_df.reset_index().drop(columns=['cycle_count'])

#                     expected_base_df.columns = [['csp_timestamp', 'v1', 'v1', 'v1', 'v2'], ['', 'COL1', 'COL2', 'COL3', '']]
#                     expected_base_df = expected_base_df[['csp_timestamp', 'v2', 'v1']]
#                     expected_base_df['csp_timestamp'] = expected_base_df['csp_timestamp'].dt.tz_localize(pytz.utc)
#                     if typ is datetime:
#                         for c in ['COL1', 'COL2', 'COL3']:
#                             expected_base_df.loc[:, ('v1', c)] = expected_base_df.loc[:, ('v1', c)].dt.tz_localize(pytz.utc)
#                     if typ is TypedCurveGenerator.SimpleEnum:
#                         for c in ['COL1', 'COL2', 'COL3']:
#                             expected_base_df.loc[:, ('v1', c)] = expected_base_df.loc[:, ('v1', c)].apply(lambda v: v.name if isinstance(v, TypedCurveGenerator.SimpleEnum) else v)
#                     if typ is TypedCurveGenerator.SimpleStruct:
#                         for k in TypedCurveGenerator.SimpleStruct.metadata():
#                             for c in ['COL1', 'COL2', 'COL3']:
#                                 expected_base_df.loc[:, (f'v1.{k}', c)] = expected_base_df.loc[:, ('v1', c)].apply(lambda v: getattr(v, k, None) if v else v)
#                         expected_base_df.drop(columns=['v1'], inplace=True, level=0)
#                     if typ is TypedCurveGenerator.NestedStruct:
#                         for k in TypedCurveGenerator.NestedStruct.metadata():
#                             for c in ['COL1', 'COL2', 'COL3']:
#                                 if k == 'value2':
#                                     expected_base_df.loc[:, (f'v1.{k}', c)] = expected_base_df.loc[:, ('v1', c)].apply(lambda v: _simple_struct_to_dict(getattr(v, k, None)) if v else v)
#                                 else:
#                                     expected_base_df.loc[:, (f'v1.{k}', c)] = expected_base_df.loc[:, ('v1', c)].apply(lambda v: getattr(v, k, None) if v else v)
#                         expected_base_df.drop(columns=['v1'], inplace=True, level=0)

#                     data_path_resolver = CacheConfigResolver(config.cache_config)
#                     base_data_df = base_graph.cached_data(data_path_resolver)().get_data_df_for_period()
#                     self.assertTrue(base_data_df.fillna(-111111).eq(expected_base_df.fillna(-111111)).all().all())
#                     missing_range_handler = lambda start, end: True
#                     start_time1 = datetime(2020, 3, 1, 9, 30, tzinfo=pytz.utc)
#                     dt1 = timedelta(hours=0, minutes=60)
#                     res1 = csp.run(run_graph, starttime=start_time1, endtime=dt1, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     self.assertEqual(len(files), 1)
#                     res1_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time1, dt1)
#                     self.assertTrue(
#                         expected_base_df[expected_base_df.csp_timestamp.between(start_time1, start_time1 + dt1)].reset_index(drop=True).fillna(-111111).eq(res1_df.fillna(-111111)).all().all())
#                     start_time2 = start_time1 + timedelta(minutes=180)
#                     dt2 = dt1
#                     res2 = csp.run(run_graph, starttime=start_time2, endtime=dt2, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     self.assertEqual(len(files), 2)
#                     res2_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time2, dt2)
#                     self.assertTrue(
#                         expected_base_df[expected_base_df.csp_timestamp.between(start_time2, start_time2 + dt2)].reset_index(drop=True).fillna(-111111).eq(res2_df.fillna(-111111)).all().all())

#                     # # Test repeated writing of the same file
#                     res2b = csp.run(run_graph, starttime=start_time2, endtime=dt2, config=config)
#                     self.assertEqual(res2b, res2)

#                     start_time3 = start_time2 + dt2 - timedelta(minutes=5)
#                     dt3 = timedelta(minutes=15)
#                     res3 = csp.run(run_graph, starttime=start_time3, endtime=dt3, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 2)
#                     res3_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time3, dt3)
#                     self.assertTrue(
#                         expected_base_df[expected_base_df.csp_timestamp.between(start_time3, start_time3 + dt3)].reset_index(drop=True).fillna(-111111).eq(res3_df.fillna(-111111)).all().all())

#                     start_time4 = start_time2 - timedelta(minutes=5)
#                     dt4 = timedelta(minutes=15)
#                     res4 = csp.run(run_graph, starttime=start_time4, endtime=dt4, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 2)
#                     res4_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time4, dt4)
#                     self.assertTrue(
#                         expected_base_df[expected_base_df.csp_timestamp.between(start_time4, start_time4 + dt4)].reset_index(drop=True).fillna(-111111).eq(res4_df.fillna(-111111)).all().all())

#                     start_time5 = start_time1 + timedelta(minutes=40)
#                     dt5 = timedelta(minutes=200)

#                     res5 = csp.run(run_graph, starttime=start_time5, endtime=dt5, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 1)
#                     res5_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time5, dt5)
#                     self.assertTrue(expected_base_df[expected_base_df.csp_timestamp.between(start_time5, start_time5 + dt5)].reset_index(drop=True).equals(res5_df))

#                     start_time6 = start_time1 + timedelta(minutes=10)
#                     res6 = csp.run(run_graph, starttime=start_time6, endtime=dt1, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period(missing_range_handler=missing_range_handler).values())
#                     if config.cache_config.merge_existing_files:
#                         self.assertEqual(len(files), 1)
#                     res6_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time6, dt1)
#                     self.assertTrue(
#                         expected_base_df[expected_base_df.csp_timestamp.between(start_time6, start_time6 + dt1)].reset_index(drop=True).fillna(-111111).eq(res6_df.fillna(-111111)).all().all())
#                     start_time7 = start_time1 - timedelta(minutes=10)
#                     dt7 = start_time3 + dt3 + timedelta(minutes=10)
#                     res7 = csp.run(run_graph, starttime=start_time7, endtime=dt7, config=config)
#                     files = list(cached_graph.cached_data(config.cache_config.data_folder)().get_data_files_for_period().values())
#                     self.assertEqual(len(files), 1)
#                     res7_df = cached_graph.cached_data(data_path_resolver)().get_data_df_for_period(start_time7, dt7)
#                     self.assertTrue(expected_base_df[expected_base_df.csp_timestamp.between(start_time7, dt7)].reset_index(drop=True).fillna(-111111).eq(res7_df.fillna(-111111)).all().all())

#     def test_subtype_dict_caching(self):
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             for cache in (True, False):
#                 @graph(cache=cache)
#                 def main() -> csp.Outputs(o=csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape=(['A', 'B']))):
#                     curve_generator = TypedCurveGenerator(period=timedelta(seconds=1))
#                     return csp.output(o={
#                         'A': curve_generator.gen_transformed_curve(TypedCurveGenerator.SimpleSubStruct, 100, 10, 1),
#                         'B': curve_generator.gen_transformed_curve(TypedCurveGenerator.SimpleSubStruct, 500, 10, 1),
#                     })

#                 start_time = datetime(2021, 1, 1)
#                 end_time = start_time + timedelta(seconds=11)
#                 if cache:
#                     with self.assertRaises(csp.impl.types.instantiation_type_resolver.ArgTypeMismatchError):
#                         csp.run(main, starttime=start_time, endtime=end_time, config=config)
#                 else:
#                     csp.run(main, starttime=start_time, endtime=end_time, config=config)

#     def test_subclass_caching(self):
#         @csp.graph
#         def main() -> csp.Outputs(o=csp.ts[TypedCurveGenerator.SimpleStruct]):
#             return csp.output(o=csp.const(TypedCurveGenerator.SimpleSubStruct()))

#         @csp.graph(cache=True)
#         def main_cached() -> csp.Outputs(o=csp.ts[TypedCurveGenerator.SimpleStruct]):
#             return csp.output(o=csp.const(TypedCurveGenerator.SimpleSubStruct()))

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(seconds=11)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(main, starttime=start_time, endtime=end_time, config=config)
#             # Cached graphs must return exact types
#             with self.assertRaises(csp.impl.types.instantiation_type_resolver.TSArgTypeMismatchError):
#                 csp.run(main_cached, starttime=start_time, endtime=end_time, config=config)

#     def test_key_subset(self):
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             @graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'tickers'}))
#             def main(tickers: [str]) -> csp.Outputs(prices=csp.OutputBasket(Dict[str, csp.ts[float]], shape="tickers")):
#                 curve_generator = TypedCurveGenerator(period=timedelta(seconds=1))
#                 return csp.output(prices={
#                     'AAPL': curve_generator.gen_transformed_curve(float, 100, 10, 1),
#                     'IBM': curve_generator.gen_transformed_curve(float, 500, 10, 1),
#                 })

#             start_time = datetime(2021, 1, 1)
#             end_time = start_time + timedelta(seconds=11)
#             res1 = csp.run(main, ['AAPL', 'IBM'], starttime=start_time, endtime=end_time, config=config)
#             res2 = csp.run(main, ['AAPL'], starttime=start_time, endtime=end_time, config=config)
#             res3 = csp.run(main, ['IBM'], starttime=start_time, endtime=end_time, config=config)
#             self.assertEqual(len(res1), 2)
#             self.assertEqual(len(res2), 1)
#             self.assertEqual(len(res3), 1)
#             self.assertEqual(res1['prices[AAPL]'], res2['prices[AAPL]'])
#             self.assertEqual(res1['prices[IBM]'], res3['prices[IBM]'])

#     def test_simple_node_caching(self):
#         throw_exc = [False]

#         @csp.node(cache=True)
#         def main_node() -> csp.Outputs(x=csp.ts[int]):
#             with csp.alarms():
#                 a = csp.alarm( int )
#             with csp.start():
#                 if throw_exc[0]:
#                     raise RuntimeError("Shouldn't get here, node should be cached")
#                 csp.schedule_alarm(a, timedelta(), 0)

#             if csp.ticked(a):
#                 csp.schedule_alarm(a, timedelta(seconds=1), a + 1)
#                 return csp.output(x=a)

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(seconds=11)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             res1 = csp.run(main_node, starttime=start_time, endtime=end_time, config=config)
#             throw_exc[0] = True
#             res2 = csp.run(main_node, starttime=start_time, endtime=end_time, config=config)
#             self.assertEqual(res1, res2)

#     def test_node_caching_with_args(self):
#         throw_exc = [False]

#         @csp.node(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'input_ts', 'input_basket'}))
#         def main_node(input_ts: csp.ts[int], input_basket: {str: csp.ts[int]}, addition: int = Injected('addition_value')) -> csp.Outputs(
#                         o1=csp.ts[int], o2=csp.OutputBasket(Dict[str, csp.ts[int]], shape_of='input_basket')):
#             with csp.alarms():
#                 a = csp.alarm( int )
#             with csp.start():
#                 if throw_exc[0]:
#                     raise RuntimeError("Shouldn't get here, node should be cached")
#                 csp.schedule_alarm(a, timedelta(), -42)
#             if csp.ticked(input_ts):
#                 csp.output(o1=input_ts + addition)
#             for k, v in input_basket.tickeditems():
#                 csp.output(o2={k: v + addition})

#         def main_graph():
#             curve_generator = TypedCurveGenerator(period=timedelta(seconds=1))
#             return main_node(curve_generator.gen_int_curve(0, 10, 1), {'1': curve_generator.gen_int_curve(10, 10, 1), '2': curve_generator.gen_int_curve(20, 10, 1)})

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(seconds=11)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             with set_new_registry_thread_instance():
#                 register_injected_object('addition_value', 42)
#                 res1 = csp.run(main_graph, starttime=start_time, endtime=end_time, config=config)
#                 throw_exc[0] = True
#                 res2 = csp.run(main_graph, starttime=start_time, endtime=end_time, config=config)
#                 self.assertEqual(res1, res2)

#     def test_caching_int_as_float(self):
#         @csp.graph(cache=True)
#         def main_cached() -> csp.Outputs(o=csp.ts[float]):
#             return csp.output(o=csp.const.using(T=int)(int(42)))

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(seconds=11)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             res1 = csp.run(main_cached, starttime=start_time, endtime=end_time, config=config)
#             res2 = csp.run(main_cached, starttime=start_time, endtime=end_time, config=config)
#             cached_val = res2['o'][0][1]
#             self.assertIs(type(cached_val), float)
#             self.assertEqual(cached_val, 42.0)

#     def test_consecutive_files_merge(self):
#         for split_columns_to_files in (True, False):
#             @csp.graph(cache=True, cache_options=GraphCacheOptions(split_columns_to_files=split_columns_to_files))
#             def main_cached() -> csp.Outputs(o=csp.ts[float]):
#                 return csp.output(o=csp.const.using(T=int)(int(42)))

#             start_time = datetime(2021, 1, 1)
#             end_time = start_time + timedelta(seconds=11)
#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 csp.run(main_cached, starttime=start_time, endtime=end_time, config=config)
#                 csp.run(main_cached, starttime=end_time + timedelta(microseconds=1), endtime=end_time + timedelta(seconds=1), config=config)
#                 files = list(main_cached.cached_data(config.cache_config.data_folder)().get_data_files_for_period().items())
#                 self.assertEqual(len(files), 1)
#                 self.assertEqual(files[0][0], (start_time, start_time + timedelta(seconds=12)))

#     def test_aggregation(self):
#         ref_date = datetime(2021, 1, 1)
#         dfs = []
#         for aggregation_period in TimeAggregation:
#             for split_columns_to_files in (True, False):
#                 @csp.node(cache=True, cache_options=GraphCacheOptions(split_columns_to_files=split_columns_to_files,
#                                                                       time_aggregation=aggregation_period))
#                 def n1() -> csp.Outputs(c=csp.ts[int]):
#                     with csp.alarms():
#                         a_t = csp.alarm( date )
#                     with csp.start():
#                         first_out_time = ref_date + timedelta(days=math.ceil((csp.now() - ref_date).total_seconds() / 86400 / 5) * 5)
#                         csp.schedule_alarm(a_t, first_out_time, ref_date.date())

#                     if csp.ticked(a_t):
#                         csp.schedule_alarm(a_t, timedelta(days=5), csp.now().date())
#                         return csp.output(c=int((csp.now() - ref_date).total_seconds() / 86400))

#                 with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                     for i in range(100):
#                         csp.run(n1, starttime=ref_date + timedelta(days=7 * i), endtime=timedelta(days=8, microseconds=1), config=config)

#                     all_parquet_files = glob.glob(os.path.join(config.cache_config.data_folder, '**', '*.parquet'), recursive=True)
#                     files_for_period = n1.cached_data(config.cache_config.data_folder)().get_data_files_for_period()
#                     dfs.append(n1.cached_data(config.cache_config.data_folder)().get_data_df_for_period())
#                     self.assertTrue((dfs[-1]['c'].diff().iloc[1:] == 5).all())
#                     num_parquet_files = len(all_parquet_files) // 2 if split_columns_to_files else len(all_parquet_files)

#                     if aggregation_period == TimeAggregation.DAY:
#                         self.assertEqual(len(files_for_period), 702)
#                         self.assertEqual(num_parquet_files, 702)
#                     elif aggregation_period == TimeAggregation.MONTH:
#                         self.assertEqual(len(files_for_period), 24)
#                         self.assertEqual(num_parquet_files, 24)
#                     elif aggregation_period == TimeAggregation.QUARTER:
#                         self.assertEqual(len(files_for_period), 8)
#                         self.assertEqual(num_parquet_files, 8)
#                     else:
#                         self.assertEqual(len(files_for_period), 2)
#                         self.assertEqual(num_parquet_files, 2)
#         for df1, df2 in zip(dfs[0:-1], dfs[1:]):
#             self.assertTrue((df1 == df2).all().all())

#     def test_struct_column_subset_read(self):
#         for split_columns_to_files in (True, False):
#             @graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'t'}, split_columns_to_files=split_columns_to_files))
#             def g(t: 'T' = TypedCurveGenerator.SimpleSubStruct) -> csp.Outputs(o=csp.ts['T']):
#                 curve_generator = TypedCurveGenerator(period=timedelta(seconds=1))
#                 return csp.output(o=curve_generator.gen_transformed_curve(t, 0, 10, 1))

#             @graph
#             def g_single_col() -> csp.Outputs(value=csp.ts[float]):

#                 return csp.output(value=g().o.value2)

#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 start_time = datetime(2021, 1, 1)
#                 end_time = start_time + timedelta(seconds=11)
#                 res1 = csp.run(g, starttime=start_time, endtime=end_time, config=config)
#                 res2 = csp.run(g.cached, TypedCurveGenerator.SimpleSubStruct, starttime=start_time, endtime=end_time, config=config)
#                 res3 = csp.run(g, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time, config=config)
#                 res4 = csp.run(g.cached, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time, config=config)
#                 # Since now we try to write with a different schema, this should raise
#                 with self.assertRaisesRegex(RuntimeError, "Metadata mismatch .*"):
#                     res5 = csp.run(g, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time + timedelta(seconds=1), config=config)

#                 self.assertEqual(res1, res2)
#                 self.assertEqual(res3, res4)
#                 self.assertEqual(len(res1['o']), len(res3['o']))
#                 self.assertNotEqual(res1, res3)
#                 for (t1, v1), (t2, v2) in zip(res1['o'], res3['o']):
#                     v1_aux = TypedCurveGenerator.SimpleStruct()
#                     v1_aux.copy_from(v1)
#                     self.assertEqual(t1, t2)
#                     self.assertEqual(v1_aux, v2)
#                 res5 = csp.run(g_single_col, starttime=start_time, endtime=end_time, config=config)
#                 files = g.cached_data(config)().get_data_files_for_period()
#                 self.assertEqual(len(files), 1)
#                 file = next(iter(files.values()))
#                 if split_columns_to_files:
#                     os.unlink(os.path.join(file, 'o.value1.parquet'))
#                 else:
#                     import pandas
#                     df = pandas.read_parquet(file)
#                     df = df.drop(columns=['o.value1'])
#                     df.to_parquet(file)

#                 res6 = csp.run(g_single_col, starttime=start_time, endtime=end_time, config=config)
#                 self.assertEqual(res5, res6)
#                 # Since we removed some data when trying to read all again, we should fail
#                 if split_columns_to_files:
#                     with self.assertRaisesRegex(Exception, 'IOError.*'):
#                         res7 = csp.run(g.cached, TypedCurveGenerator.SimpleSubStruct, starttime=start_time, endtime=end_time, config=config)
#                 else:
#                     with self.assertRaisesRegex(RuntimeError, '.*Missing column o.value1.*'):
#                         res7 = csp.run(g.cached, TypedCurveGenerator.SimpleSubStruct, starttime=start_time, endtime=end_time, config=config)

#     def test_basket_struct_column_subset_read(self):
#         @graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'t'}))
#         def g(t: 'T' = TypedCurveGenerator.SimpleSubStruct) ->  csp.Outputs(o=csp.OutputBasket(Dict[str, csp.ts['T']], shape=['my_key'])) :
#             curve_generator = TypedCurveGenerator(period=timedelta(seconds=1))
#             return csp.output(o={'my_key': curve_generator.gen_transformed_curve(t, 0, 10, 1)})

#         @graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'t'}))
#         def g_unnamed_out(t: 'T' = TypedCurveGenerator.SimpleSubStruct) -> csp.OutputBasket(Dict[str, csp.ts['T']], shape=['my_key']):
#             return g.cached(t).o

#         @graph
#         def g_single_col(unnamed: bool = False) -> csp.Outputs(value=csp.ts[float]):

#             if unnamed:
#                 res = csp.get_basket_field(g_unnamed_out(), 'value2')
#             else:
#                 res = csp.get_basket_field(g().o, 'value2')

#             return csp.output(value=res['my_key'])

#         def verify_all(x: csp.ts[bool]):
#             self.assertTrue(x is not None)

#         @graph
#         def g_verify_multiple_type():
#             verify_all(g_unnamed_out(TypedCurveGenerator.SimpleStruct)['my_key'].value2 == g_unnamed_out(TypedCurveGenerator.SimpleSubStruct)['my_key'].value2)

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             start_time = datetime(2021, 1, 1)
#             end_time = start_time + timedelta(seconds=11)
#             res1 = csp.run(g, starttime=start_time, endtime=end_time, config=config)
#             res2 = csp.run(g.cached, TypedCurveGenerator.SimpleSubStruct, starttime=start_time, endtime=end_time, config=config)
#             res3 = csp.run(g, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time, config=config)
#             res4 = csp.run(g.cached, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time, config=config)
#             res_unnamed = csp.run(g_unnamed_out, starttime=start_time, endtime=end_time, config=config)
#             res_unnamed_cached = csp.run(g_unnamed_out.cached, starttime=start_time, endtime=end_time, config=config)
#             csp.run(g_verify_multiple_type, starttime=start_time, endtime=end_time, config=config)
#             self.assertEqual(res_unnamed, res_unnamed_cached)
#             self.assertEqual(res_unnamed['my_key'], res1['o[my_key]'])
#             # Since now we try to write with a different schema, this should raise
#             with self.assertRaisesRegex(RuntimeError, "Metadata mismatch .*"):
#                 res5 = csp.run(g, TypedCurveGenerator.SimpleStruct, starttime=start_time, endtime=end_time + timedelta(seconds=1), config=config)

#             self.assertEqual(res1, res2)
#             self.assertEqual(res3, res4)
#             self.assertEqual(len(res1['o[my_key]']), len(res3['o[my_key]']))
#             self.assertNotEqual(res1, res3)
#             for (t1, v1), (t2, v2) in zip(res1['o[my_key]'], res3['o[my_key]']):
#                 v1_aux = TypedCurveGenerator.SimpleStruct()
#                 v1_aux.copy_from(v1)
#                 self.assertEqual(t1, t2)
#                 self.assertEqual(v1_aux, v2)
#             res5 = csp.run(g_single_col, False, starttime=start_time, endtime=end_time, config=config)
#             res5_unnamed = csp.run(g_single_col, True, starttime=start_time, endtime=end_time, config=config)
#             self.assertEqual(res5, res5_unnamed)
#             files = g.cached_data(config)().get_data_files_for_period()
#             self.assertEqual(len(files), 1)
#             file = next(iter(files.values()))
#             # TODO: uncomment
#             # os.unlink(os.path.join(file, 'o.value1.parquet'))

#             res6 = csp.run(g_single_col, starttime=start_time, endtime=end_time, config=config)
#             self.assertEqual(res5, res6)
#             # Since we removed some data when trying to read all again, we should fail
#             # TODO: uncomment
#             # with self.assertRaisesRegex(Exception, 'IOError.*'):
#             #     res7 = csp.run(g.cached, TypedCurveGenerator.SimpleSubStruct, starttime=start_time, endtime=end_time, config=config)

#     def test_unnamed_output_caching(self):
#         for split_columns_to_files in (True, False):
#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(split_columns_to_files=split_columns_to_files))
#                 def g_scalar() -> csp.ts[int]:
#                     gen = TypedCurveGenerator()
#                     return gen.gen_int_curve(0, 10, 1)

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(split_columns_to_files=split_columns_to_files))
#                 def g_struct() -> csp.ts[TypedCurveGenerator.SimpleStruct]:
#                     gen = TypedCurveGenerator()
#                     return gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 0, 10, 1)

#                 @csp.graph(cache=True)
#                 def g_scalar_basket() -> csp.OutputBasket(Dict[str, csp.ts[int]] , shape=['k1', 'k2']):
#                     gen = TypedCurveGenerator()
#                     return {'k1': gen.gen_int_curve(0, 10, 1),
#                             'k2': gen.gen_int_curve(100, 10, 1)}

#                 @csp.graph(cache=True)
#                 def g_struct_basket() -> csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape=['k1', 'k2']):
#                     gen = TypedCurveGenerator()
#                     return {'k1': gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 0, 10, 1),
#                             'k2': gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 100, 10, 1)}

#                 def run_test_single_graph(g_func):
#                     res1 = csp.run(g_func, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                    config=config)
#                     res2 = csp.run(g_func, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                    config=config)
#                     res3 = csp.run(g_func.cached, starttime=datetime(2020, 3, 1, 20, 30), endtime=timedelta(hours=0, minutes=390),
#                                    config=config)
#                     self.assertEqual(res1, res2)
#                     self.assertEqual(res2, res3)
#                     return res1

#                 run_test_single_graph(g_scalar)
#                 run_test_single_graph(g_struct)
#                 run_test_single_graph(g_scalar_basket)
#                 run_test_single_graph(g_struct_basket)

#                 res1_df = g_scalar.cached_data(config)().get_data_df_for_period()
#                 res2_df = g_struct.cached_data(config)().get_data_df_for_period()
#                 res3_df = g_scalar_basket.cached_data(config)().get_data_df_for_period()
#                 res4_df = g_struct_basket.cached_data(config)().get_data_df_for_period()
#                 self.assertEqual(list(res1_df.columns), ['csp_timestamp', 'csp_unnamed_output'])
#                 self.assertEqual(list(res2_df.columns), ['csp_timestamp', 'value1', 'value2'])
#                 self.assertEqual(list(res3_df.columns), ['csp_timestamp', 'k1', 'k2'])
#                 self.assertEqual(list(res4_df.columns), [('csp_timestamp', ''), ('value1', 'k1'), ('value1', 'k2'), ('value2', 'k1'), ('value2', 'k2')])

#                 for df in (res1_df, res2_df, res3_df, res4_df):
#                     self.assertEqual(len(df), 11)

#     def test_basket_ids_retrieval(self):
#         for aggregation_period in (TimeAggregation.MONTH, TimeAggregation.DAY,):
#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'keys'}, time_aggregation=aggregation_period))
#                 def g(keys: object) -> csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape="keys"):
#                     gen = TypedCurveGenerator()
#                     return {keys[0]: gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 0, 10, 1),
#                             keys[1]: gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 100, 10, 1)}

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'keys'}))
#                 def g_named_output(keys: object) -> csp.Outputs(out=csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape="keys")):

#                     gen = TypedCurveGenerator()
#                     return csp.output(out={keys[0]: gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 0, 10, 1),
#                                     keys[1]: gen.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, 100, 10, 1)})

#                 csp.run(g, ['k1', 'k2'], starttime=datetime(2020, 3, 1), endtime=datetime(2020, 3, 1, 23, 59, 59, 999999),
#                         config=config)
#                 csp.run(g, ['k3', 'k4'], starttime=datetime(2020, 3, 2), endtime=datetime(2020, 3, 2, 23, 59, 59, 999999),
#                         config=config)
#                 csp.run(g_named_output, ['k1', 'k2'], starttime=datetime(2020, 3, 1), endtime=datetime(2020, 3, 1, 23, 59, 59, 999999),
#                         config=config)
#                 csp.run(g_named_output, ['k3', 'k4'], starttime=datetime(2020, 3, 2), endtime=datetime(2020, 3, 2, 23, 59, 59, 999999),
#                         config=config)
#                 self.assertEqual(g.cached_data(config)().get_all_basket_ids_in_range(), ['k1', 'k2', 'k3', 'k4'])
#                 self.assertEqual(g.cached_data(config)().get_all_basket_ids_in_range(starttime=datetime(2020, 3, 1), endtime=datetime(2020, 3, 1, 23, 59, 59, 999999)),
#                                  ['k1', 'k2'])
#                 self.assertEqual(g.cached_data(config)().get_all_basket_ids_in_range(starttime=datetime(2020, 3, 2), endtime=datetime(2020, 3, 2, 23, 59, 59, 999999)),
#                                  ['k3', 'k4'])
#                 self.assertEqual(g_named_output.cached_data(config)().get_all_basket_ids_in_range('out'), ['k1', 'k2', 'k3', 'k4'])
#                 self.assertEqual(g_named_output.cached_data(config)().get_all_basket_ids_in_range('out', starttime=datetime(2020, 3, 1), endtime=datetime(2020, 3, 1, 23, 59, 59, 999999)),
#                                  ['k1', 'k2'])
#                 self.assertEqual(g_named_output.cached_data(config)().get_all_basket_ids_in_range('out', starttime=datetime(2020, 3, 2), endtime=datetime(2020, 3, 2, 23, 59, 59, 999999)),
#                                  ['k3', 'k4'])

#     def test_custom_time_fields(self):
#         from csp.impl.wiring.graph import NoCachedDataException
#         import numpy

#         @csp.graph(cache=True, cache_options=GraphCacheOptions(data_timestamp_column_name='timestamp'))
#         def g1() -> csp.ts[_DummyStructWithTimestamp]:
#             s = csp.engine_start_time()
#             return csp.curve(_DummyStructWithTimestamp, [(s + timedelta(hours=1 + i),
#                                                           _DummyStructWithTimestamp(val=i, timestamp=s + timedelta(hours=(2 * i) ** 2))) for i in range(10)])

#         @csp.graph(cache=True, cache_options=GraphCacheOptions(data_timestamp_column_name='timestamp'))
#         def g2() -> csp.Outputs(timestamp=csp.ts[datetime], values=csp.OutputBasket(Dict[str, csp.ts[int]], shape=['v1', 'v2'])):
#             s = csp.engine_start_time()
#             values = {}
#             values['v1'] = csp.curve(int, [(s + timedelta(hours=1 + i), i) for i in range(10)])
#             values['v2'] = csp.curve(int, [(s + timedelta(hours=1 + i), i * 100) for i in range(10)])
#             t = csp.curve(datetime, [(s + timedelta(hours=1 + i), s + timedelta(hours=(2 * i) ** 2)) for i in range(10)])
#             return csp.output(timestamp=t, values=values)

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             s = datetime(2021, 1, 1)
#             csp.run(g1, starttime=s, endtime=timedelta(hours=100), config=config)
#             csp.run(g2, starttime=s, endtime=timedelta(hours=100), config=config)
#             data_files = g1.cached_data(config)().get_data_files_for_period()
#             data_files2 = g2.cached_data(config)().get_data_files_for_period()
#             res = csp.run(g1.cached, starttime=datetime(2021, 1, 1), endtime=datetime(2021, 1, 14, 12, 0), config=config)
#             res2 = csp.run(g2.cached, starttime=datetime(2021, 1, 1), endtime=datetime(2021, 1, 14, 12, 0), config=config)
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(g1.cached, starttime=datetime(2021, 1, 1), endtime=datetime(2021, 1, 14, 12, 1), config=config)

#             self.assertEqual(list(data_files.keys()), list(data_files2.keys()))
#             self.assertEqual([(k, v.val) for k, v in res[0]], res2['values[v1]'])
#             all_file_time_ranges = list(data_files.keys())
#             expected_start_end = res[0][0][1].timestamp, res[0][-1][1].timestamp
#             actual_start_end = all_file_time_ranges[0][0], all_file_time_ranges[-1][1]
#             self.assertEqual(expected_start_end, actual_start_end)
#             data_df = g1.cached_data(config)().get_data_df_for_period()
#             data_df2 = g2.cached_data(config)().get_data_df_for_period()
#             self.assertTrue(all((data_df.timestamp.diff().dt.total_seconds() / 3600).values[1:].astype(int) == numpy.diff(((numpy.arange(0, 10) * 2) ** 2))))
#             self.assertTrue(all(data_df.val.values == (numpy.arange(0, 10))))
#             self.assertTrue((data_df['val'] == data_df2['values']['v1']).all())
#             self.assertTrue((data_df['val'] * 100 == data_df2['values']['v2']).all())

#     def test_cached_with_start_stop_times(self):
#         @csp.graph(cache=True)
#         def g() -> csp.ts[int]:
#             return csp.curve(int, [(datetime(2021, 1, 1), 1), (datetime(2021, 1, 2), 2), (datetime(2021, 1, 3), 3)])

#         @csp.graph
#         def g2(csp_cache_start: object = None) -> csp.ts[int]:
#             end = csp.engine_end_time() - timedelta(days=1, microseconds=-1)
#             if csp_cache_start:
#                 cached_g = g.cached[csp_cache_start:end]
#             else:
#                 cached_g = g.cached[:end]
#             return csp.delay(cached_g(), timedelta(days=1))

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             res1 = csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             res2 = csp.run(g2, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1) + timedelta(days=1), config=config)
#             res1_transformed = [(v1 + timedelta(days=1), v2) for (v1, v2) in res1[0]]
#             self.assertEqual(res1_transformed, res2[0])
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(g2, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1) + timedelta(days=1, microseconds=1), config=config)

#             res3 = csp.run(g2, datetime(2021, 1, 1, 1), starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1) + timedelta(days=1), config=config)
#             self.assertEqual(res3[0], res2[0][1:])

#     def test_cached_graph_not_instantiated(self):
#         raise_exception = [False]

#         @csp.graph(cache=True)
#         def g() -> csp.ts[int]:
#             assert not raise_exception[0]
#             return csp.curve(int, [(datetime(2021, 1, 1), 1), (datetime(2021, 1, 2), 2), (datetime(2021, 1, 3), 3)])

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             raise_exception[0] = True
#             csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(g.cached, starttime=datetime(2021, 1, 1), endtime=timedelta(days=4, microseconds=-1), config=config)

#     def test_caching_with_struct_arguments(self):
#         @csp.graph(cache=True)
#         def g(value: TypedCurveGenerator.SimpleStruct) -> csp.ts[TypedCurveGenerator.SimpleStruct]:
#             return csp.curve(TypedCurveGenerator.SimpleStruct, [(datetime(2021, 1, 1), value)])

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             s = TypedCurveGenerator.SimpleStruct(value1=42)
#             res = csp.run(g, s, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             res2 = csp.run(g.cached, s, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             self.assertEqual(res, res2)

#     def test_caching_user_types(self):
#         class OrderedDictSerializer(CacheObjectSerializer):
#             def serialize_to_bytes(self, value):
#                 import pickle
#                 return pickle.dumps(value)

#             def deserialize_from_bytes(self, value):
#                 import pickle
#                 return pickle.loads(value)

#         @csp.graph(cache=True)
#         def g() -> csp.ts[collections.OrderedDict]:
#             return csp.curve(collections.OrderedDict, [(datetime(2021, 1, 1), collections.OrderedDict({1: 2, 3: 4}))])

#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             # We don't know how to serialize ordereddict, this should raise
#             with self.assertRaises(TypeError):
#                 res = csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)

#             config.cache_config.cache_serializers[collections.OrderedDict] = OrderedDictSerializer()
#             res = csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             res2 = csp.run(g.cached, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#             self.assertEqual(res, res2)
#             res_df = g.cached_data(config)().get_data_df_for_period()
#             self.assertEqual(res_df['csp_unnamed_output'].iloc[0], res[0][0][1])

#     def test_special_character_partitioning(self):
#         # Since we're using glob to locate the files on disk, there was a bug that special characters in the partition values broke the partition data
#         # lookup. This test tests that it works now.

#         for split_columns_to_files in (True, False):
#             graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_files)

#             @csp.graph(cache=True, **graph_kwargs)
#             def g(x1: str, x2: str) -> csp.ts[str]:
#                 return csp.curve(str, [(datetime(2021, 1, 1), x1), (datetime(2021, 1, 1) + timedelta(seconds=1), x2)])

#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 x1 = "[][]"
#                 x2 = "*x*)("
#                 res = csp.run(g, x1, x2, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#                 res2 = csp.run(g.cached, x1, x2, starttime=datetime(2021, 1, 1), endtime=timedelta(days=3, microseconds=-1), config=config)
#                 self.assertEqual(res, res2)
#                 df = g.cached_data(config)(x1, x2).get_data_df_for_period()
#                 self.assertEqual(df['csp_unnamed_output'].tolist(), [x1, x2])

#     def test_cutoff_bug(self):
#         """Test for bug that was there of +-1 micro second offset, that caused some stitch data to be missing
#         :return:
#         """
#         for split_columns_to_files in (True, False):
#             if split_columns_to_files:
#                 cache_options = GraphCacheOptions(split_columns_to_files=True)
#             else:
#                 cache_options = GraphCacheOptions(split_columns_to_files=True)
#             cache_options.time_aggregation = TimeAggregation.MONTH

#             @csp.graph(cache=True, cache_options=cache_options)
#             def g() -> csp.ts[int]:
#                 l = [(datetime(2021, 1, 1), 1), (datetime(2021, 1, 1, 23, 59, 59, 999999), 2),
#                      (datetime(2021, 1, 2), 3), (datetime(2021, 1, 2, 23, 59, 59, 999999), 4),
#                      (datetime(2021, 1, 3), 5), (datetime(2021, 1, 3, 23, 59, 59, 999999), 6)]
#                 l = [v for v in l if csp.engine_start_time() <= v[0] <= csp.engine_end_time()]
#                 return csp.curve(int, l)

#             with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#                 csp.run(g, starttime=datetime(2021, 1, 1), endtime=datetime(2021, 1, 1, 23, 59, 59, 999999), config=config)
#                 csp.run(g, starttime=datetime(2021, 1, 1, 12), endtime=datetime(2021, 1, 3, 23, 59, 59, 999999), config=config)
#                 self.assertEqual(g.cached_data(config)().get_data_df_for_period()['csp_unnamed_output'].tolist(), [1, 2, 3, 4, 5, 6])

#     def test_scalar_flat_basket_loading(self):
#         @csp.graph(cache=True)
#         def simple_cached() ->  csp.Outputs(i=csp.OutputBasket(Dict[str, csp.ts[int]], shape=['V1', 'V2']),
#                                             s=csp.OutputBasket(Dict[str, csp.ts[str]], shape=['V3', 'V4'])):

#             i_v1 = csp.curve(int, [(timedelta(hours=10), 1), (timedelta(hours=10), 1), (timedelta(hours=30), 1)])
#             i_v2 = csp.curve(int, [(timedelta(hours=10), 10), (timedelta(hours=20), 11)])
#             s_v3 = csp.curve(str, [(timedelta(hours=30), "val1")])
#             s_v4 = csp.curve(str, [(timedelta(hours=10), "val2"), (timedelta(hours=20), "val3")])
#             return csp.output(i={'V1': i_v1, 'V2': i_v2}, s={'V3': s_v3, 'V4': s_v4})

#         @csp.graph(cache=True)
#         def simple_cached_unnamed() -> csp.OutputBasket(Dict[str, csp.ts[int]], shape=['V1', 'V2']):

#             return csp.output(simple_cached().i)

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(hours=30)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(simple_cached, starttime=start_time, endtime=end_time, config=config)
#             csp.run(simple_cached_unnamed, starttime=start_time, endtime=end_time, config=config)
#             df_ref_full = simple_cached.cached_data(config)().get_data_df_for_period().stack(dropna=False).reset_index().drop(columns=['level_0']).rename(columns={'level_1': 'symbol'})
#             df_ref_full['csp_timestamp'] = df_ref_full['csp_timestamp'].ffill().dt.tz_localize(None)
#             df_ref_full = df_ref_full[df_ref_full.symbol.str.len() > 0].reset_index(drop=True)

#             for start_dt, end_dt in ((None, None),
#                                      (timedelta(hours=10), None),
#                                      (timedelta(hours=10), timedelta(hours=10, microseconds=1)),
#                                      (timedelta(hours=10), timedelta(hours=20)),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=20)),
#                                      (timedelta(hours=10, microseconds=1), None),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=10, microseconds=2)),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=30))):

#                 cur_start = start_time + start_dt if start_dt else None
#                 cur_end = start_time + end_dt if end_dt else None
#                 mask = df_ref_full.index >= 0
#                 if cur_start:
#                     mask &= df_ref_full.csp_timestamp >= cur_start
#                 if cur_end:
#                     mask &= df_ref_full.csp_timestamp <= cur_end
#                 df_ref = df_ref_full[mask]

#                 df_ref_i = df_ref[['csp_timestamp', 'symbol', 'i']][~df_ref.i.isna()].reset_index(drop=True)
#                 df_ref_s = df_ref[['csp_timestamp', 'symbol', 's']][~df_ref.s.isna()].reset_index(drop=True)

#                 i_df_flat = simple_cached.cached_data(config)().get_flat_basket_df_for_period(basket_field_name='i', symbol_column='symbol',
#                                                                                               starttime=cur_start, endtime=cur_end)
#                 s_df_flat = simple_cached.cached_data(config)().get_flat_basket_df_for_period(basket_field_name='s', symbol_column='symbol',
#                                                                                               starttime=cur_start, endtime=cur_end)
#                 unnamed_flat = simple_cached_unnamed.cached_data(config)().get_flat_basket_df_for_period(symbol_column='symbol',
#                                                                                                          starttime=cur_start, endtime=cur_end)
#                 self.assertTrue((i_df_flat == df_ref_i).all().all())
#                 self.assertTrue((s_df_flat == df_ref_s).all().all())

#                 # We can't rename columns when None is returned so we have to add this check
#                 if unnamed_flat is None:
#                     self.assertTrue(len(df_ref_i) == 0)
#                 else:
#                     self.assertTrue((unnamed_flat.rename(columns={'csp_unnamed_output': 'i'}) == df_ref_i).all().all())

#     def test_struct_flat_basket_loading(self):
#         @csp.graph(cache=True)
#         def simple_cached() -> csp.Outputs(ret=csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleSubStruct]], shape=['V1', 'V2'])):

#             i_v1 = csp.curve(int, [(timedelta(hours=10), 1), (timedelta(hours=10), 1), (timedelta(hours=30), 1)])
#             i_v2 = csp.curve(int, [(timedelta(hours=10), 10), (timedelta(hours=20), 11)])
#             s_v3 = csp.curve(str, [(timedelta(hours=30), "val1")])
#             s_v4 = csp.curve(str, [(timedelta(hours=10), "val2"), (timedelta(hours=20), "val3")])
#             res = {}
#             res['V1'] = TypedCurveGenerator.SimpleSubStruct.fromts(value1=i_v1, value3=s_v3)
#             res['V2'] = TypedCurveGenerator.SimpleSubStruct.fromts(value1=i_v2, value3=s_v4)
#             return csp.output(ret=res)

#         @csp.graph(cache=True)
#         def simple_cached_unnamed() -> csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleSubStruct]], shape=['V1', 'V2']):
#             return csp.output(simple_cached().ret)

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(hours=30)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(simple_cached, starttime=start_time, endtime=end_time, config=config)
#             csp.run(simple_cached_unnamed, starttime=start_time, endtime=end_time, config=config)
#             df_ref_full = simple_cached.cached_data(config)().get_data_df_for_period().stack(dropna=False).reset_index().drop(columns=['level_0']).rename(columns={'level_1': 'symbol'})
#             df_ref_full['csp_timestamp'] = df_ref_full['csp_timestamp'].ffill().dt.tz_localize(None)
#             mask = (df_ref_full.symbol.str.len() > 0) & (~df_ref_full['ret.value1'].isna() | ~df_ref_full['ret.value2'].isna() | ~df_ref_full['ret.value2'].isna())
#             df_ref_full = df_ref_full[mask].reset_index(drop=True)
#             df_ref_full = df_ref_full[['csp_timestamp', 'symbol', 'ret.value1', 'ret.value2', 'ret.value3']]

#             for start_dt, end_dt in ((None, None),
#                                      (timedelta(hours=10), None),
#                                      (timedelta(hours=10), timedelta(hours=10, microseconds=1)),
#                                      (timedelta(hours=10), timedelta(hours=20)),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=20)),
#                                      (timedelta(hours=10, microseconds=1), None),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=10, microseconds=2)),
#                                      (timedelta(hours=10, microseconds=1), timedelta(hours=30))):

#                 cur_start = start_time + start_dt if start_dt else None
#                 cur_end = start_time + end_dt if end_dt else None
#                 mask = df_ref_full.index >= 0
#                 if cur_start:
#                     mask &= df_ref_full.csp_timestamp >= cur_start
#                 if cur_end:
#                     mask &= df_ref_full.csp_timestamp <= cur_end
#                 df_ref = df_ref_full[mask].fillna(-999).reset_index(drop=True)

#                 df_flat = simple_cached.cached_data(config)().get_flat_basket_df_for_period(basket_field_name='ret', symbol_column='symbol',
#                                                                                             starttime=cur_start, endtime=cur_end)

#                 unnamed_flat = simple_cached_unnamed.cached_data(config)().get_flat_basket_df_for_period(symbol_column='symbol',
#                                                                                                          starttime=cur_start, endtime=cur_end)
#                 if unnamed_flat is not None:
#                     unnamed_flat_normalized = unnamed_flat.rename(columns=dict(zip(unnamed_flat.columns, df_flat.columns)))
#                 if df_flat is None:
#                     self.assertTrue(len(df_ref) == 0)
#                     self.assertTrue(unnamed_flat is None)
#                 else:
#                     self.assertTrue((df_flat.fillna(-999) == df_ref.fillna(-999)).all().all())
#                     self.assertTrue((df_flat.fillna(-999) == unnamed_flat_normalized.fillna(-999)).all().all())

#                 for c in TypedCurveGenerator.SimpleSubStruct.metadata().keys():
#                     df_flat_single_col = simple_cached.cached_data(config)().get_flat_basket_df_for_period(basket_field_name='ret', symbol_column='symbol', struct_fields=[c],
#                                                                                                            starttime=cur_start, endtime=cur_end)
#                     if df_flat_single_col is None:
#                         self.assertTrue(len(df_ref) == 0)
#                         continue
#                     df_flat_single_col_ref = df_ref[df_flat_single_col.columns]
#                     self.assertTrue((df_flat_single_col.fillna(-999) == df_flat_single_col_ref.fillna(-999)).all().all())

#     def test_simple_time_shift(self):
#         @csp.graph(cache=True)
#         def simple_cached() -> csp.ts[int]:

#             return csp.curve(int, [(timedelta(hours=i), i) for i in range(72)])

#         @csp.graph
#         def cached_data_shifted(shift: timedelta) -> csp.ts[int]:
#             return simple_cached.cached.shifted(csp_timestamp_shift=shift)()

#         def to_df(res):
#             return pandas.DataFrame({'timestamp': res[0][0], 'value': res[0][1]})

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(hours=71)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(simple_cached, starttime=start_time, endtime=end_time, config=config)
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(cached_data_shifted, timedelta(minutes=1), starttime=start_time, endtime=end_time, config=config, output_numpy=True)

#             ref_df = to_df(csp.run(cached_data_shifted, timedelta(), starttime=start_time, endtime=end_time, config=config, output_numpy=True))
#             td12 = timedelta(hours=12)

#             shifted_df = ref_df.shift(12).iloc[12:, :].reset_index(drop=True)
#             shifted_df['timestamp'] += td12
#             res_df1 = to_df(csp.run(cached_data_shifted, td12, starttime=start_time + td12, endtime=end_time, config=config, output_numpy=True))
#             res_df2 = to_df(csp.run(cached_data_shifted, td12, starttime=start_time + td12, endtime=end_time + td12, config=config, output_numpy=True))
#             self.assertTrue((shifted_df == res_df1).all().all())
#             self.assertTrue((ref_df.value == res_df2.value).all())

#     def test_struct_basket_time_shift(self):
#         @csp.graph(cache=True)
#         def struct_cached() -> csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape=['A', 'B']):

#             generator = TypedCurveGenerator(period=timedelta(hours=1))
#             return {
#                 'A': generator.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, start_value=0, num_cycles=71, increment=1, duplicate_timestamp_indices=[11], skip_indices=[3, 15]),
#                 'B': generator.gen_transformed_curve(TypedCurveGenerator.SimpleStruct, start_value=100, num_cycles=71, increment=1, duplicate_timestamp_indices=[10, 11, 12, 13, 14],
#                                                      skip_indices=[3, 15])
#             }

#         @csp.node
#         def dict_builder(x: {str: csp.ts[TypedCurveGenerator.SimpleStruct]}) -> csp.ts[object]:
#             res = {'timestamp': csp.now()}
#             ticked_items = {k: v for k, v in x.tickeditems()}
#             for k in x.keys():
#                 res[k] = ticked_items.get(k)
#             return res

#         @csp.graph
#         def cached_data_shifted(shift: timedelta) -> csp.ts[object]:
#             return dict_builder(struct_cached.cached.shifted(csp_timestamp_shift=shift)())

#         def to_df(res):
#             keys = list(res[0][1][0].keys())
#             values = [[v for k, v in d.items()] for d in res[0][1]]
#             return pandas.DataFrame(dict(zip(keys, zip(*values))))

#         start_time = datetime(2021, 1, 1)
#         end_time = start_time + timedelta(hours=71)
#         with _GraphTempCacheFolderConfig(allow_overwrite=True) as config:
#             csp.run(struct_cached, starttime=start_time, endtime=end_time, config=config)
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(cached_data_shifted, timedelta(minutes=1), starttime=start_time, endtime=end_time, config=config, output_numpy=True)

#             ref_df = to_df(csp.run(cached_data_shifted, timedelta(), starttime=start_time, endtime=end_time, config=config, output_numpy=True))
#             td12 = timedelta(hours=12)

#             ref_df1 = ref_df.copy()
#             ref_df1.timestamp += td12
#             ref_df1 = ref_df1[ref_df1.timestamp.between(start_time + td12, end_time)].reset_index(drop=True)
#             res_df1 = to_df(csp.run(cached_data_shifted, td12, starttime=start_time + td12, endtime=end_time, config=config, output_numpy=True))
#             self.assertTrue((res_df1.fillna(-1) == ref_df1.fillna(-1)).all().all())

#             ref_df2 = ref_df.copy()
#             ref_df2.timestamp += td12
#             ref_df2 = ref_df2[ref_df2.timestamp.between(start_time + td12, end_time + td12)].reset_index(drop=True)
#             res_df2 = to_df(csp.run(cached_data_shifted, td12, starttime=start_time + td12, endtime=end_time + td12, config=config, output_numpy=True))
#             self.assertTrue((ref_df2.fillna(-1) == res_df2.fillna(-1)).all().all())

#             ref_df2 = ref_df.copy()
#             ref_df2.timestamp -= td12
#             ref_df2 = ref_df2[ref_df2.timestamp.between(start_time - td12, end_time - td12)].reset_index(drop=True)
#             res_df2 = to_df(csp.run(cached_data_shifted, -td12, starttime=start_time - td12, endtime=end_time - td12, config=config, output_numpy=True))
#             self.assertTrue((ref_df2.fillna(-1) == res_df2.fillna(-1)).all().all())

#     def test_caching_separate_folder(self):
#         @csp.graph(cache=True)
#         def g(name: str) -> csp.ts[float]:
#             if name == 'a':
#                 return csp.curve(float, [(timedelta(seconds=i), i) for i in range(10)])
#             else:
#                 return csp.curve(float, [(timedelta(seconds=i), i * 2) for i in range(10)])

#         with _GraphTempCacheFolderConfig() as config:
#             with _GraphTempCacheFolderConfig() as config2:
#                 res1 = csp.run(g, 'a', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                 # Write to a different folder data for a different day and key
#                 res2 = csp.run(g, 'a', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config2)
#                 res2b = csp.run(g, 'b', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config2)

#                 config3 = config.copy()
#                 files1 = g.cached_data(config3)('a').get_data_files_for_period()
#                 self.assertEqual(len(files1), 1)

#                 csp.run(g.cached, 'a', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config)
#                 with self.assertRaises(NoCachedDataException):
#                     csp.run(g.cached, 'a', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config)
#                 with self.assertRaises(NoCachedDataException):
#                     csp.run(g.cached, 'b', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config)

#                 config3.cache_config.read_folders = [config2.cache_config.data_folder]
#                 files2 = g.cached_data(config3)('a').get_data_files_for_period(missing_range_handler=lambda *args, **kwargs: True)
#                 self.assertEqual(len(files2), 2)

#                 res3 = csp.run(g, 'a', starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=30), config=config2)
#                 res4 = csp.run(g, 'a', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config2)
#                 res4b = csp.run(g, 'b', starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=30), config=config2)
#                 self.assertEqual(res1, res3)
#                 self.assertEqual(res2, res4)
#                 self.assertEqual(res2b, res4b)

#     def test_cache_invalidation(self):
#         for split_columns_to_file in (True, False):
#             with _GraphTempCacheFolderConfig() as config:
#                 graph_kwargs = self._get_default_graph_caching_kwargs(split_columns_to_file)

#             @csp.graph(cache=True, **graph_kwargs)
#             def my_graph(val: str) -> csp.ts[float]:
#                 return csp.curve(float, [(timedelta(days=i), float(i)) for i in range(5)])

#             start1 = datetime(2021, 1, 1)
#             end1 = start1 + timedelta(days=5, microseconds=-1)

#             self.assertTrue(my_graph.cached_data(config) is None)
#             csp.run(my_graph, 'val1', starttime=start1, endtime=end1, config=config)
#             # We should be able to invalidate cache when no cached data exists yet
#             my_graph.cached_data(config)('val2').invalidate_cache()
#             csp.run(my_graph, 'val2', starttime=start1, endtime=end1, config=config)
#             cached_data1 = my_graph.cached_data(config)('val1').get_data_df_for_period()
#             cached_data2 = my_graph.cached_data(config)('val2').get_data_df_for_period()
#             self.assertTrue((cached_data1 == cached_data2).all().all())
#             self.assertEqual(len(cached_data1), 5)
#             my_graph.cached_data(config)('val2').invalidate_cache(start1 + timedelta(days=1), end1)
#             cached_data2_after_invalidation = my_graph.cached_data(config)('val2').get_data_df_for_period()
#             self.assertTrue((cached_data1.head(1) == cached_data2_after_invalidation).all().all())
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(my_graph.cached, 'val2', starttime=start1, endtime=end1, config=config)
#             # this should run fine, we still have data
#             csp.run(my_graph.cached, 'val2', starttime=start1, endtime=start1 + timedelta(days=1, microseconds=-1), config=config)
#             my_graph.cached_data(config)('val2').invalidate_cache()
#             # now we have no data
#             with self.assertRaises(NoCachedDataException):
#                 csp.run(my_graph.cached, 'val2', starttime=start1, endtime=start1 + timedelta(days=1, microseconds=-1), config=config)
#             my_graph.cached_data(config)('val1').invalidate_cache()
#             self.assertTrue(my_graph.cached_data(config)('val1').get_data_df_for_period() is None)
#             # We should still be able to invalidate
#             my_graph.cached_data(config)('val1').invalidate_cache()
#             # We should have no data in the data folder
#             self.assertFalse(os.listdir(os.path.join(my_graph.cached_data(config)._dataset.data_paths.root_folder, 'data')))

#     def test_controlled_cache(self):
#         for default_cache_enabled in (True, False):
#             with _GraphTempCacheFolderConfig() as config:
#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def graph_unnamed_output1() -> csp.ts[float]:
#                     csp.set_cache_enable_ts(csp.curve(bool, [(timedelta(seconds=5), True), (timedelta(seconds=6.1), False), (timedelta(seconds=8), True)]))

#                     return csp.output(csp.curve(float, [(timedelta(seconds=i), float(i)) for i in range(10)]))

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def graph_unnamed_output2() -> csp.ts[float]:
#                     csp.set_cache_enable_ts(csp.curve(bool, [(timedelta(seconds=5), True), (timedelta(seconds=6.1), False), (timedelta(seconds=8), True)]))

#                     return (csp.curve(float, [(timedelta(seconds=i), float(i)) for i in range(10)]))

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def graph_single_named_output() -> csp.Outputs(res1=csp.ts[float]):
#                     csp.set_cache_enable_ts(csp.curve(bool, [(timedelta(seconds=5), True), (timedelta(seconds=6.1), False), (timedelta(seconds=8), True)]))

#                     return csp.output(res1=csp.curve(float, [(timedelta(seconds=i), float(i)) for i in range(10)]))

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def graph_multiple_outputs() -> csp.Outputs(res1=csp.ts[float], res2=csp.OutputBasket(Dict[str, csp.ts[float]], shape=['value'])):
#                     csp.set_cache_enable_ts(csp.curve(bool, [(timedelta(seconds=5), True), (timedelta(seconds=6.1), False), (timedelta(seconds=8), True)]))

#                     res = csp.curve(float, [(timedelta(seconds=i), float(i)) for i in range(10)])
#                     return csp.output(res1=res, res2={'value': res})

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def main_graph_cached_named_output() -> csp.Outputs(res1=csp.ts[float]):
#                     csp.set_cache_enable_ts(csp.curve(bool, [(timedelta(seconds=5), True), (timedelta(seconds=6.1), False), (timedelta(seconds=8), True)]))

#                     return csp.output(csp.curve(float, [(timedelta(seconds=i), float(i)) for i in range(10)]))

#                 @csp.node(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def node_unnamed_output() -> csp.ts[float]:
#                     with csp.alarms():
#                         a_enable = csp.alarm( bool )
#                         a_value = csp.alarm( float )
#                     with csp.start():
#                         csp.schedule_alarm(a_enable, timedelta(seconds=5), True)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=6.1), False)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=8), True)
#                         csp.schedule_alarm(a_value, timedelta(), 0)
#                     if csp.ticked(a_enable):
#                         csp.enable_cache(a_enable)
#                     if csp.ticked(a_value):
#                         if a_value < 9:
#                             csp.schedule_alarm(a_value, timedelta(seconds=1), a_value + 1)
#                         if a_value == 6:
#                             return a_value
#                         elif a_value == 8:
#                             return csp.output(a_value)
#                             raise NotImplementedError()
#                         else:
#                             csp.output(a_value)

#                 @csp.node(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def node_single_named_output() -> csp.Outputs(res=csp.ts[float]):
#                     with csp.alarms():
#                         a_enable = csp.alarm( bool )
#                         a_value = csp.alarm( float )
#                     with csp.start():
#                         csp.schedule_alarm(a_enable, timedelta(seconds=5), True)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=6.1), False)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=8), True)
#                         csp.schedule_alarm(a_value, timedelta(), 0)
#                     if csp.ticked(a_enable):
#                         csp.enable_cache(a_enable)
#                     if csp.ticked(a_value):
#                         if a_value < 9:
#                             csp.schedule_alarm(a_value, timedelta(seconds=1), a_value + 1)
#                         if a_value == 6:
#                             return a_value
#                         elif a_value == 8:
#                             return csp.output(res=a_value)
#                             raise NotImplementedError()
#                         else:
#                             csp.output(res=a_value)

#                 @csp.node(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def node_multiple_outputs() ->  csp.Outputs(res1=csp.ts[float], res2=csp.OutputBasket(Dict[str, csp.ts[float]], shape=['value'])):
#                     with csp.alarms():
#                         a_enable = csp.alarm( bool )
#                         a_value = csp.alarm( float )
#                     with csp.start():
#                         csp.schedule_alarm(a_enable, timedelta(seconds=5), True)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=6.1), False)
#                         csp.schedule_alarm(a_enable, timedelta(seconds=8), True)
#                         csp.schedule_alarm(a_value, timedelta(), 0)
#                     if csp.ticked(a_enable):
#                         csp.enable_cache(a_enable)
#                     if csp.ticked(a_value):
#                         if a_value < 9:
#                             csp.schedule_alarm(a_value, timedelta(seconds=1), a_value + 1)
#                         csp.output(res2={'value': a_value})
#                         return csp.output(res1=a_value)
#                         raise NotImplementedError()

#                 starttime = datetime(2020, 1, 1, 9, 29)
#                 endtime = starttime + timedelta(minutes=20)

#                 results = []
#                 for g in [graph_unnamed_output1, graph_unnamed_output2, graph_single_named_output, graph_multiple_outputs, node_unnamed_output,
#                           node_single_named_output, node_multiple_outputs]:
#                     csp.run(g, starttime=starttime, endtime=endtime, config=config)
#                     results.append(g.cached_data(config)().get_data_df_for_period(missing_range_handler=lambda *a, **ka: True))
#                 for res in results:
#                     if default_cache_enabled:
#                         self.assertEqual(len(res), 9)
#                     else:
#                         self.assertEqual(len(res), 4)
#                 combined_df = pandas.concat(results[:1] + [res.drop(columns=['csp_timestamp']) for res in results], axis=1)
#                 self.assertEqual(list(combined_df.columns),
#                                  ['csp_timestamp', 'csp_unnamed_output', 'csp_unnamed_output', 'csp_unnamed_output', 'res1',
#                                   ('res1', ''), ('res2', 'value'), 'csp_unnamed_output', 'res', ('res1', ''), ('res2', 'value')])
#                 self.assertTrue((combined_df.iloc[:, 1:].diff(axis=1).iloc[:, 1:] == 0).all().all())

#     def test_controlled_cache_never_set(self):
#         """
#         Test that if we never output the controolled set control, we don't get any errors
#         :return:
#         """
#         for default_cache_enabled in (True, False):
#             with _GraphTempCacheFolderConfig() as config:
#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def graph_unnamed_output1() -> csp.ts[float]:
#                     return csp.output(csp.null_ts(float))

#                 @csp.node(cache=True, cache_options=GraphCacheOptions(controlled_cache=True, default_cache_enabled=default_cache_enabled))
#                 def node_unnamed_output() -> csp.ts[float]:
#                     with csp.alarms():
#                         a_enable = csp.alarm( bool )
#                         a_value = csp.alarm( float )
#                     if False:
#                         csp.output(a_value)

#                 starttime = datetime(2020, 1, 1, 9, 29)
#                 endtime = starttime + timedelta(minutes=20)

#                 csp.run(graph_unnamed_output1, starttime=starttime, endtime=endtime, config=config)

#     def test_controlled_cache_bug(self):
#         """
#         There was a bug when we run across multiple aggregation periods that the cache enabled was not handled properly, this test was written to reproduce the bug
#         and fix it. Here we have aggregation period of 1 month but we are running across 2 months.
#         :return:
#         """
#         @csp.graph(
#             cache=True,
#             cache_options=GraphCacheOptions(
#                 controlled_cache=True,
#                 time_aggregation=TimeAggregation.MONTH))
#         def cached_g() -> csp.ts[int]:

#             csp.set_cache_enable_ts(csp.curve(bool, [(datetime(2004, 7, 1), True), (datetime(2004, 8, 2), False)]))
#             return csp.curve(int, [(datetime(2004, 6, 30), 1),
#                                    (datetime(2004, 7, 1), 2),
#                                    (datetime(2004, 7, 2), 3),
#                                    (datetime(2004, 8, 1), 4),
#                                    (datetime(2004, 8, 1, 1), 5),
#                                    (datetime(2004, 8, 2), 6),
#                                    ])

#         @csp.graph(
#             cache=True,
#             cache_options=GraphCacheOptions(
#                 controlled_cache=True,
#                 time_aggregation=TimeAggregation.MONTH))
#         def cached_g_struct() -> csp.ts[_DummyStructWithTimestamp]:

#             csp.set_cache_enable_ts(csp.curve(bool, [(datetime(2004, 7, 1), True), (datetime(2004, 8, 2), False)]))
#             return _DummyStructWithTimestamp.fromts(val=cached_g())

#         @csp.graph(
#             cache=True,
#             cache_options=GraphCacheOptions(
#                 controlled_cache=True,
#                 time_aggregation=TimeAggregation.MONTH))
#         def cached_g_basket() -> csp.OutputBasket(Dict[str, csp.ts[int]], shape=['a', 'b']):

#             csp.set_cache_enable_ts(csp.curve(bool, [(datetime(2004, 7, 1), True), (datetime(2004, 8, 2), False)]))
#             return {'a': csp.curve(int, [(datetime(2004, 6, 30), 1),
#                                          (datetime(2004, 7, 1), 2),
#                                          (datetime(2004, 7, 2), 3),
#                                          (datetime(2004, 8, 1), 4),
#                                          (datetime(2004, 8, 1, 1), 5),
#                                          (datetime(2004, 8, 2), 6),
#                                          ]),
#                     'b': csp.curve(int, [(datetime(2004, 6, 30), 1), (datetime(2004, 8, 1, 1), 5), ])
#                     }

#         @csp.graph(
#             cache=True,
#             cache_options=GraphCacheOptions(
#                 controlled_cache=True,
#                 time_aggregation=TimeAggregation.MONTH))
#         def cached_g_basket_struct() -> csp.OutputBasket(Dict[str, csp.ts[_DummyStructWithTimestamp]], shape=['a', 'b']):

#             csp.set_cache_enable_ts(csp.curve(bool, [(datetime(2004, 7, 1), True), (datetime(2004, 8, 2), False)]))
#             aux = cached_g_basket()
#             return {'a': _DummyStructWithTimestamp.fromts(val=aux['a']),
#                     'b': _DummyStructWithTimestamp.fromts(val=aux['b'])}

#         def g():
#             cached_g()
#             cached_g_struct()
#             cached_g_basket()
#             cached_g_basket_struct()

#         with _GraphTempCacheFolderConfig() as config:
#             starttime = datetime(2004, 6, 30)
#             endtime = datetime(2004, 8, 2, 23, 59, 59, 999999)
#             csp.run(g, starttime=starttime, endtime=endtime, config=config)
#             df = cached_g.cached_data(config)().get_data_df_for_period()
#             self.assertEqual(df.csp_unnamed_output.tolist(), [2, 3, 4, 5])
#             struct_df = cached_g_struct.cached_data(config)().get_data_df_for_period()
#             self.assertEqual(struct_df.val.tolist(), [2, 3, 4, 5])
#             basket_df = cached_g_basket.cached_data(config)().get_data_df_for_period()
#             self.assertEqual(basket_df.a.tolist(), [2, 3, 4, 5])
#             self.assertEqual(basket_df.b.fillna(-1).tolist(), [-1, -1, -1, 5])
#             struct_basket_df = cached_g_basket_struct.cached_data(config)().get_data_df_for_period()
#             self.assertEqual(struct_basket_df.val.a.tolist(), [2, 3, 4, 5])
#             self.assertEqual(struct_basket_df.val.b.fillna(-1).tolist(), [-1, -1, -1, 5])

#     def test_numpy_1d_array_caching(self):
#         for split_columns_to_files in (True, False):
#             cache_args = self._get_default_graph_caching_kwargs(split_columns_to_files=split_columns_to_files)

#             for typ in (int, bool, float, str):
#                 a1 = numpy.array([1, 2, 3, 4, 0], dtype=typ)
#                 a2 = numpy.array([[1, 2], [3324, 4]], dtype=typ)[:, 0]
#                 self.assertTrue(a1.flags.c_contiguous)
#                 self.assertFalse(a2.flags.c_contiguous)

#                 @csp.graph(cache=True, **cache_args)
#                 def g1() -> csp.ts[csp.typing.Numpy1DArray[typ]]:
#                     return csp.flatten([csp.const(a1), csp.const(a2)])

#                 @csp.graph(cache=True, cache_options=GraphCacheOptions(parquet_output_config=ParquetOutputConfig(batch_size=3),
#                                                                        split_columns_to_files=split_columns_to_files))
#                 def g2() -> csp.ts[csp.typing.Numpy1DArray[typ]]:
#                     return csp.flatten([csp.const(a1), csp.const(a2)])

#                 @csp.node(cache=True, **cache_args)
#                 def n1() -> csp.Outputs(arr1=csp.ts[csp.typing.Numpy1DArray[typ]], arr2=csp.ts[numpy.ndarray]):
#                     with csp.alarms():
#                         a_values1 = csp.alarm( csp.typing.Numpy1DArray )
#                         a_values2 = csp.alarm( numpy.ndarray )

#                     with csp.start():
#                         csp.schedule_alarm(a_values1, timedelta(0), a1)
#                         csp.schedule_alarm(a_values1, timedelta(seconds=1), a2)
#                         csp.schedule_alarm(a_values2, timedelta(0), numpy.array([numpy.nan, 1]))
#                         csp.schedule_alarm(a_values2, timedelta(0), numpy.array([2, numpy.nan, 3]))

#                     if csp.ticked(a_values1):
#                         csp.output(arr1=a_values1)
#                     if csp.ticked(a_values2):
#                         csp.output(arr2=a_values2)

#                 def verify_equal_array(expected_list, result):
#                     res_list = [v for t, v in result[0]]
#                     self.assertEqual(len(expected_list), len(res_list))
#                     for e, r in zip(expected_list, res_list):
#                         self.assertTrue((e == r).all())

#                 def verify_n1_result(expected_list, result):
#                     verify_equal_array(expected_list, {0: result['arr1']})
#                     arr2_values = [v for _, v in result['arr2']]
#                     expected_arr2_values = numpy.array([numpy.array([numpy.nan, 1.]), numpy.array([2., numpy.nan, 3.])], dtype=object)
#                     self.assertEqual(len(arr2_values), len(expected_arr2_values))
#                     for v1, v2 in zip(arr2_values, expected_arr2_values):
#                         self.assertTrue(((v1 == v2) | (numpy.isnan(v1) & (numpy.isnan(v1) == numpy.isnan(v1)))).all())

#                 with _GraphTempCacheFolderConfig() as config:
#                     starttime = datetime(2020, 1, 1, 9, 29)
#                     endtime = starttime + timedelta(minutes=20)
#                     expected_list = [a1, a2]

#                     res = csp.run(g1, starttime=starttime, endtime=endtime, config=config)
#                     verify_equal_array(expected_list, res)
#                     res = csp.run(g1.cached, starttime=starttime, endtime=endtime, config=config)
#                     verify_equal_array(expected_list, res)
#                     res = csp.run(n1, starttime=starttime, endtime=endtime, config=config)
#                     verify_n1_result(expected_list, res)
#                     res = csp.run(n1.cached, starttime=starttime, endtime=endtime, config=config)
#                     verify_n1_result(expected_list, res)
#                     res = csp.run(g2, starttime=starttime, endtime=endtime, config=config)
#                     verify_equal_array(expected_list, res)
#                     res = csp.run(g2.cached, starttime=starttime, endtime=endtime, config=config)
#                     verify_equal_array(expected_list, res)

#     def test_numpy_wrong_type_errors(self):
#         @csp.graph(cache=True)
#         def g1() -> csp.ts[csp.typing.Numpy1DArray[int]]:
#             return csp.const(numpy.zeros(1, dtype=float))

#         @csp.graph(cache=True)
#         def g2() -> csp.ts[csp.typing.Numpy1DArray[object]]:
#             return csp.const(numpy.zeros(1, dtype=object))

#         @csp.node(cache=True)
#         def n1() -> csp.ts[csp.typing.Numpy1DArray[int]]:
#             with csp.alarms():
#                 a_out = csp.alarm( bool )
#             with csp.start():
#                 csp.schedule_alarm(a_out, timedelta(), True)
#             if csp.ticked(a_out):
#                 return numpy.zeros(1, dtype=float)

#         with _GraphTempCacheFolderConfig() as config:
#             with self.assertRaisesRegex(TSArgTypeMismatchError, re.escape("In function g1: Expected ts[csp.typing.Numpy1DArray[int]] for return value, got ts[csp.typing.Numpy1DArray[float]]")):
#                 csp.run(g1, starttime=datetime(2020, 1, 1), endtime=timedelta(minutes=20), config=config)

#         with _GraphTempCacheFolderConfig() as config:
#             with self.assertRaisesRegex(TypeError, re.escape("Unsupported array value type when writing to parquet:DIALECT_GENERIC")):
#                 csp.run(g2, starttime=datetime(2020, 1, 1), endtime=timedelta(minutes=20), config=config)

#         with _GraphTempCacheFolderConfig() as config:
#             with self.assertRaisesRegex(TypeError, re.escape("Expected array of type dtype('int64') got dtype('float64')")):
#                 csp.run(n1, starttime=datetime(2020, 1, 1), endtime=timedelta(minutes=20), config=config)

#     def test_basket_array_caching(self):
#         @csp.graph(cache=True)
#         def g1() -> csp.OutputBasket(Dict[str, csp.ts[csp.typing.Numpy1DArray[int]]], shape=['a']):
#             a = numpy.zeros(3, dtype=int)
#             return {
#                 'a': csp.const(a)
#             }

#         with _GraphTempCacheFolderConfig() as config:
#             starttime = datetime(2020, 1, 1, 9, 29)
#             endtime = starttime + timedelta(minutes=20)

#             with self.assertRaisesRegex(NotImplementedError, re.escape('Writing of baskets with array values is not supported')):
#                 res = csp.run(g1, starttime=starttime, endtime=endtime, config=config)

#     def test_multi_dimensional_array_caching(self):
#         a1 = numpy.array([1, 2, 3, 4, 0], dtype=float)
#         a2 = numpy.array([[1, 2], [3, 4]], dtype=float)
#         expected_df = pandas.DataFrame.from_dict({'csp_timestamp': [pytz.utc.localize(datetime(2020, 1, 1, 9, 29))] * 2, 'csp_unnamed_output': [a1, a2]})
#         for split_columns_to_files in (True, False):
#             cache_args = self._get_default_graph_caching_kwargs(split_columns_to_files=split_columns_to_files)

#             @csp.graph(cache=True, **cache_args)
#             def g1() -> csp.ts[csp.typing.NumpyNDArray[float]]:
#                 return csp.flatten([csp.const(a1), csp.const(a2)])

#             @csp.graph(cache=True, **cache_args)
#             def g2() -> csp.Outputs(res=csp.ts[csp.typing.NumpyNDArray[float]]):
#                 return csp.output(res=g1())

#             with _GraphTempCacheFolderConfig() as config:
#                 starttime = datetime(2020, 1, 1, 9, 29)
#                 endtime = starttime + timedelta(minutes=20)
#                 csp.run(g2, starttime=starttime, endtime=endtime, config=config)
#                 res_cached = csp.run(g1.cached, starttime=starttime, endtime=endtime, config=config)
#                 df = g1.cached_data(config)().get_data_df_for_period()
#                 df2 = g2.cached_data(config)().get_data_df_for_period()
#                 self.assertEqual(len(df), len(expected_df))
#                 self.assertTrue((df.csp_timestamp == expected_df.csp_timestamp).all())
#                 self.assertTrue(all([(v1 == v2).all() for v1, v2 in zip(df['csp_unnamed_output'], expected_df['csp_unnamed_output'])]))
#                 cached_values = list(zip(*res_cached[0]))[1]
#                 self.assertTrue(all([(v1 == v2).all() for v1, v2 in zip(cached_values, expected_df['csp_unnamed_output'])]))
#                 # We need to check the named column as well.
#                 self.assertEqual(len(df2), len(expected_df))
#                 self.assertTrue(all([(v1 == v2).all() for v1, v2 in zip(df2['res'], expected_df['csp_unnamed_output'])]))

#     def test_read_folder_data_load_as_df(self):
#         @csp.graph(cache=True)
#         def g1() -> csp.ts[float]:
#             return csp.const(42.0)

#         with _GraphTempCacheFolderConfig() as config1:
#             with _GraphTempCacheFolderConfig() as config2:
#                 starttime = datetime(2020, 1, 1, 9, 29)
#                 endtime = starttime + timedelta(minutes=20)
#                 csp.run(g1, starttime=starttime, endtime=endtime, config=config1)
#                 res1_cached = g1.cached_data(config1)().get_data_df_for_period()
#                 config2.cache_config.read_folders = [config1.cache_config.data_folder]
#                 res2_cached = g1.cached_data(config2)().get_data_df_for_period()
#                 self.assertTrue((res1_cached == res2_cached).all().all())

#     def test_multiple_readers_different_shapes(self):
#         @csp.graph(cache=True, cache_options=GraphCacheOptions(ignored_inputs={'shape', 'dummy'}))
#         def g(shape: [str], dummy: object) -> csp.OutputBasket(Dict[str, csp.ts[str]], shape="shape"):
#             res = {}
#             for v in shape:
#                 res[v] = csp.const(f'{v}_value')
#             return res

#         @csp.graph
#         def read_g():
#             __outputs__(v1={str: csp.ts[str]}, v2={str: csp.ts[str]})
#             df = pandas.DataFrame({'dummy': [1]})

#             v2_a = g.cached(['b', 'c', 'd'], df)
#             v2_b = g.cached(['b', 'c', 'd'], df)
#             assert id(v2_a) == id(v2_b)

#             return csp.output(v1=g.cached(['a', 'b'], df), v2=v2_a)

#         with _GraphTempCacheFolderConfig() as config:
#             starttime = datetime(2020, 1, 1, 9, 29)
#             endtime = starttime + timedelta(minutes=1)
#             csp.run(g, ['a', 'b', 'c', 'd'], None, starttime=starttime, endtime=endtime, config=config)
#             res = csp.run(read_g, starttime=starttime, endtime=endtime, config=config)
#             self.assertEqual(sorted(res.keys()), ['v1[a]', 'v1[b]', 'v2[b]', 'v2[c]', 'v2[d]'])

#     def test_basket_partial_cache_load(self):
#         @csp.graph(cache=True)
#         def g1() -> csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape=['A', 'B']):
#             return {'A': csp.curve(TypedCurveGenerator.SimpleStruct, [(timedelta(seconds=0), TypedCurveGenerator.SimpleStruct(value1=0)),
#                                                                       (timedelta(seconds=1), TypedCurveGenerator.SimpleStruct(value1=2)),
#                                                                       ]),
#                     'B': csp.curve(TypedCurveGenerator.SimpleStruct, [(timedelta(seconds=1), TypedCurveGenerator.SimpleStruct(value1=3)),
#                                                                       (timedelta(seconds=2), TypedCurveGenerator.SimpleStruct(value1=4)),
#                                                                       ])
#                     }

#         @csp.graph(cache=True)
#         def g2() -> csp.Outputs(my_named_output=csp.OutputBasket(Dict[str, csp.ts[TypedCurveGenerator.SimpleStruct]], shape=['A', 'B'])):
#             return csp.output(my_named_output=g1())

#         with _GraphTempCacheFolderConfig() as config:
#             starttime = datetime(2020, 1, 1, 9, 29)
#             endtime = starttime + timedelta(minutes=1)
#             csp.run(g2, starttime=starttime, endtime=endtime, config=config)
#             res1 = g1.cached_data(config)().get_data_df_for_period()
#             res2 = g1.cached_data(config)().get_data_df_for_period(struct_basket_sub_columns={'': ['value1']})
#             with self.assertRaisesRegex(RuntimeError, re.escape("Specified sub columns for basket 'csp_unnamed_output' but it's not loaded from file") + '.*'):
#                 res3 = g2.cached_data(config)().get_data_df_for_period(struct_basket_sub_columns={'': ['value1']})
#             res3 = g2.cached_data(config)().get_data_df_for_period(struct_basket_sub_columns={'my_named_output': ['value1']})

#             self.assertEqual(res1.columns.levels[0].to_list(), ['csp_timestamp', 'value1', 'value2'])
#             self.assertEqual(res1.columns.levels[1].to_list(), ['', 'A', 'B'])

#             self.assertEqual(res2.columns.levels[0].to_list(), ['csp_timestamp', 'value1'])
#             self.assertEqual(res2.columns.levels[1].to_list(), ['', 'A', 'B'])

#             self.assertEqual(res3.columns.levels[0].to_list(), ['csp_timestamp', 'my_named_output.value1'])
#             self.assertEqual(res3.columns.levels[1].to_list(), ['', 'A', 'B'])

#             self.assertTrue((res1['value1'].fillna(-111111) == res2['value1'].fillna(-111111)).all().all())
#             self.assertTrue((res1['value1'].fillna(-111111) == res3['my_named_output.value1'].fillna(-111111)).all().all())
#             self.assertTrue((res1['csp_timestamp'] == res2['csp_timestamp']).all().all())
#             self.assertTrue((res1['csp_timestamp'] == res3['csp_timestamp']).all().all())

#     def test_partition_retrieval(self):
#         @csp.graph(cache=True)
#         def g1(i: int, d: date, dt: datetime, td: timedelta, f_val: float, s: str, b: bool, struct: TypedCurveGenerator.SimpleStruct) -> csp.Outputs(v1=csp.ts[TypedCurveGenerator.SimpleStruct], v2=csp.ts[float]):
#             return csp.output(v1=csp.const(struct), v2=csp.const(f_val))

#         @csp.graph(cache=True)
#         def g2() -> csp.ts[int]:
#             return csp.const(42)

#         s1 = datetime(2020, 1, 1)
#         e1 = s1 + timedelta(hours=70, microseconds=-1)
#         with _GraphTempCacheFolderConfig() as config:
#             # i: int, d: date, dt: datetime, td: timedelta, f: float, s: str, b: bool, struct: TypedCurveGenerator.SimpleStruct
#             csp.run(g1, i=1, d=date(2013, 5, 8), dt=datetime(2025, 3, 6, 11, 20, 59, 999599), td=timedelta(seconds=5), f_val=5.3, s="test1", b=False,
#                     struct=TypedCurveGenerator.SimpleStruct(value1=53),
#                     starttime=s1, endtime=e1, config=config)
#             csp.run(g1, i=52, d=date(2013, 5, 31), dt=datetime(2025, 3, 5), td=timedelta(days=100), f_val=7.8, s="test2", b=True, struct=TypedCurveGenerator.SimpleStruct(value1=-53), starttime=s1,
#                     endtime=e1, config=config)
#             csp.run(g2, starttime=s1, endtime=e1, config=config)
#             g1_keys = g1.cached_data(config).get_partition_keys()
#             g2_keys = g2.cached_data(config).get_partition_keys()
#             self.assertEqual([DatasetPartitionKey({'i': 1,
#                                                    'd': date(2013, 5, 8),
#                                                    'dt': datetime(2025, 3, 6, 11, 20, 59, 999599),
#                                                    'td': timedelta(seconds=5),
#                                                    'f_val': 5.3,
#                                                    's': 'test1',
#                                                    'b': False,
#                                                    'struct': TypedCurveGenerator.SimpleStruct(value1=53)}),
#                               DatasetPartitionKey({'i': 52,
#                                                    'd': date(2013, 5, 31),
#                                                    'dt': datetime(2025, 3, 5),
#                                                    'td': timedelta(days=100),
#                                                    'f_val': 7.8,
#                                                    's': 'test2',
#                                                    'b': True,
#                                                    'struct': TypedCurveGenerator.SimpleStruct(value1=-53)})],
#                              g1_keys)
#             # self.assertEqual(len(g1_keys), 2)
#             self.assertEqual([DatasetPartitionKey({})], g2_keys)
#             df1 = g1.cached_data(config)(**g1_keys[0].kwargs).get_data_df_for_period()
#             df2 = g2.cached_data(config)(**g2_keys[0].kwargs).get_data_df_for_period()
#             self.assertTrue((df1.fillna(-42) == pandas.DataFrame({'csp_timestamp': [pytz.utc.localize(s1)], 'v1.value1': [53], 'v1.value2': [-42], 'v2': [5.3]})).all().all())
#             self.assertTrue((df2 == pandas.DataFrame({'csp_timestamp': [pytz.utc.localize(s1)], 'csp_unnamed_output': [42]})).all().all())


# if __name__ == '__main__':
#     unittest.main()
