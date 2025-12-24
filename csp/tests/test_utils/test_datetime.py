from csp.utils.datetime import utc_now


def test_utc_now():
    now = utc_now()
    assert now.tzinfo is None
