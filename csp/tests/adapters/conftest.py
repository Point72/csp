import pytest

from csp.adapters.kafka import KafkaAdapterManager


@pytest.fixture(scope="module", autouse=True)
def kafkabroker():
    return "localhost:9092"


@pytest.fixture(scope="module", autouse=True)
def kafkaadapter(kafkabroker):
    group_id = "group.id123"
    _kafkaadapter = KafkaAdapterManager(
        broker=kafkabroker, group_id=group_id, rd_kafka_conf_options={"allow.auto.create.topics": "true"}
    )
    return _kafkaadapter
