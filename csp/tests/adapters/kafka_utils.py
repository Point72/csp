import httpx


def _precreate_topic(topic):
    """Since we test against confluent kafka, just use the kafka rest addon"""
    rest_broker = "http://localhost:8082"
    cluster_info = httpx.get(f"{rest_broker}/v3/clusters")
    cluster_id = cluster_info.json()["data"][0]["cluster_id"]
    resp = httpx.post(f"{rest_broker}/v3/clusters/{cluster_id}/topics", json={"topic_name": topic})
    if resp.status_code != 201 and "already exists" not in resp.content.decode("utf8"):
        raise Exception(
            f"Could not create topic {topic} on cluster {rest_broker}/v3/clusters/{cluster_id}/topics - received {resp.content}"
        )
