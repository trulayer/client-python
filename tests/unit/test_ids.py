
from trulayer._ids import new_id


def test_new_id_returns_string() -> None:
    id_ = new_id()
    assert isinstance(id_, str)
    assert len(id_) == 36  # UUID format


def test_new_id_is_unique() -> None:
    assert new_id() != new_id()


def test_new_id_is_valid_uuid_format() -> None:
    id_ = new_id()
    parts = id_.split("-")
    assert len(parts) == 5
    assert len(parts[0]) == 8
