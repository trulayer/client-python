import warnings
from unittest.mock import patch

import httpx
import pytest
import respx

import trulayer
from trulayer.client import TruLayerClient


def _make_client(**kwargs: object) -> TruLayerClient:
    with patch("trulayer.batch.BatchSender.start"):
        return TruLayerClient(api_key="tl_test", project_id="proj-1", **kwargs)  # type: ignore[arg-type]


def test_trace_returns_context() -> None:
    client = _make_client()
    ctx = client.trace("my-trace")
    assert ctx._data.project_id == "proj-1"
    assert ctx._data.name == "my-trace"


def test_trace_with_tags_and_metadata() -> None:
    client = _make_client()
    ctx = client.trace(tags=["test"], metadata={"env": "ci"})
    assert ctx._data.tags == ["test"]
    assert ctx._data.metadata == {"env": "ci"}


@respx.mock
def test_feedback_success() -> None:
    respx.post("https://api.trulayer.ai/v1/feedback").mock(
        return_value=httpx.Response(201, json={"id": "fb-1"})
    )
    client = _make_client()
    # Should not raise
    client.feedback(trace_id="trace-1", label="good", score=1.0)


@respx.mock
def test_feedback_failure_warns_not_raises() -> None:
    respx.post("https://api.trulayer.ai/v1/feedback").mock(return_value=httpx.Response(500))
    client = _make_client()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client.feedback(trace_id="trace-1", label="bad")
        assert any("feedback" in str(warning.message).lower() for warning in w)


def test_global_init_and_get_client() -> None:
    with patch("trulayer.batch.BatchSender.start"):
        c = trulayer.init(api_key="tl_test", project_id="proj-global")
    assert trulayer.get_client() is c


def test_get_client_before_init_raises() -> None:
    import trulayer as tl

    tl._global_client = None
    with pytest.raises(RuntimeError, match="init"):
        tl.get_client()


def test_sample_rate_default_is_1() -> None:
    client = _make_client()
    assert client._sample_rate == 1.0


def test_sample_rate_stored_on_client() -> None:
    client = _make_client(sample_rate=0.1)
    assert client._sample_rate == 0.1


def test_sample_rate_invalid_raises() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        _make_client(sample_rate=1.5)

    with pytest.raises(ValueError, match="sample_rate"):
        _make_client(sample_rate=-0.1)


def test_init_passes_sample_rate() -> None:
    with patch("trulayer.batch.BatchSender.start"):
        c = trulayer.init(api_key="tl_test", project_id="proj-1", sample_rate=0.25)
    assert c._sample_rate == 0.25


def test_scrub_fn_default_is_none() -> None:
    client = _make_client()
    assert client._scrub_fn is None


def test_scrub_fn_stored_on_client() -> None:
    fn = lambda s: s.upper()  # noqa: E731
    client = _make_client(scrub_fn=fn)
    assert client._scrub_fn is fn


def test_init_passes_scrub_fn() -> None:
    fn = lambda s: "[REDACTED]"  # noqa: E731
    with patch("trulayer.batch.BatchSender.start"):
        c = trulayer.init(api_key="tl_test", project_id="proj-1", scrub_fn=fn)
    assert c._scrub_fn is fn


def test_init_accepts_deprecated_project_id_alias() -> None:
    import warnings

    import trulayer

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = trulayer.TruLayerClient(api_key="tl_t", project_id="legacy")
    assert c._project_name == "legacy"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    c.shutdown(timeout=0.1)


def test_init_requires_project_name() -> None:
    import pytest

    import trulayer

    with pytest.raises(TypeError, match="project_name"):
        trulayer.TruLayerClient(api_key="tl_t")
