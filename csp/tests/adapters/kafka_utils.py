__all__ = ("create_topic",)


def create_topic(broker, topic):
    """Create `topic` and block until the broker acknowledges it.

    Creation is done out of band rather than by publishing a warm-up message, so that no test data
    lands on the topic and so tests do not depend on broker-side auto-creation (disabled in
    ci/kafka/docker-compose.yml so test_invalid_topic can exercise the failure path).
    """
    # Imported lazily so collection does not require confluent-kafka when the kafka tests are skipped
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": broker})
    for _, future in admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)]).items():
        future.result()
