import pytest

from csp.adapters.kafka import KafkaAdapterManager


@pytest.fixture(scope="module", autouse=True)
def kafkabroker():
    # Defined in ci/kafka/docker-compose.yml
    return "localhost:9092"


@pytest.fixture(scope="module", autouse=True)
def kafkaadapter(kafkabroker):
    group_id = "group.id123"
    _kafkaadapter = KafkaAdapterManager(broker=kafkabroker, group_id=group_id)
    return _kafkaadapter
