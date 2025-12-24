from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Get the current UTC time, removes tzinfo to keep consistent with csp graph expectations and current datetime.utcnow() behavior.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
