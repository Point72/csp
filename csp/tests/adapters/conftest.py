from uuid import uuid4

import pytest

from csp.adapters.kafka import KafkaAdapterManager


@pytest.fixture(scope="module", autouse=True)
def kafkabroker():
    # Defined in ci/kafka/docker-compose.yml. Not "localhost": that also resolves to ::1, where
    # the container publishes nothing, and librdkafka alternates between the resolved addresses.
    return "127.0.0.1:9092"


@pytest.fixture(scope="module", autouse=True)
def kafkaadapterkwargs(kafkabroker):
    # Unique group id so a rerun never inherits committed offsets from a previous run
    return dict(broker=kafkabroker, group_id=f"csp.test.{uuid4()}")


@pytest.fixture(scope="module", autouse=True)
def kafkaadapter(kafkaadapterkwargs):
    return KafkaAdapterManager(**kafkaadapterkwargs)
