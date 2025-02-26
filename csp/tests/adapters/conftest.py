import pytest

from csp.adapters.kafka import KafkaAdapterManager


@pytest.fixture(scope="module", autouse=True)
def kafkabroker():
    return "localhost:9092"


@pytest.fixture(scope="module", autouse=True)
def kafkaadapterkwargs(kafkabroker):
    return dict(broker=kafkabroker, group_id="group.id123", rd_kafka_conf_options={"allow.auto.create.topics": "true"})


@pytest.fixture(scope="module", autouse=True)
def kafkaadapter(kafkaadapterkwargs):
    _kafkaadapter = KafkaAdapterManager(**kafkaadapterkwargs)
    return _kafkaadapter
