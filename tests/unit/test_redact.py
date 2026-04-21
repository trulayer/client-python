"""Unit tests for trulayer.redact."""

from __future__ import annotations

import re

import pytest

from trulayer.redact import BUILTIN_PACKS, Redactor, Rule, redact

# ---------------------------------------------------------------------------
# Built-in pack coverage
# ---------------------------------------------------------------------------


def test_standard_pack_catches_email() -> None:
    r = Redactor(packs=["standard"])
    assert r.redact("contact foo.bar+baz@example.co.uk today") == (
        "contact <REDACTED:email> today"
    )


def test_standard_pack_catches_ssn() -> None:
    r = Redactor(packs=["standard"])
    assert r.redact("SSN 123-45-6789.") == "SSN <REDACTED:ssn>."


def test_standard_pack_catches_bearer_token() -> None:
    r = Redactor(packs=["standard"])
    out = r.redact("Authorization: Bearer abc.DEF-123_xyz")
    assert "<REDACTED:bearer_token>" in out
    assert "abc.DEF-123_xyz" not in out


def test_standard_pack_catches_jwt() -> None:
    r = Redactor(packs=["standard"])
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc-123_XYZ"
    out = r.redact(f"token={jwt}")
    assert "<REDACTED:jwt>" in out
    assert jwt not in out


def test_standard_pack_catches_phone() -> None:
    r = Redactor(packs=["standard"])
    out = r.redact("call +1 415-555-0199 now")
    assert "<REDACTED:phone>" in out


def test_standard_pack_leaves_benign_text_alone() -> None:
    r = Redactor(packs=["standard"])
    benign = "The quick brown fox jumps over the lazy dog."
    assert r.redact(benign) == benign


def test_strict_pack_catches_ipv4() -> None:
    r = Redactor(packs=["strict"])
    assert r.redact("host 10.0.0.25 down") == "host <REDACTED:ipv4> down"


def test_strict_pack_catches_valid_credit_card_only() -> None:
    r = Redactor(packs=["strict"])
    # 4539 1488 0343 6467 is a Luhn-valid test number.
    out = r.redact("card 4539 1488 0343 6467 and noise 1234 5678 9012 3456")
    assert "<REDACTED:credit_card>" in out
    assert "1234 5678 9012 3456" in out  # not Luhn-valid — must not be redacted


def test_strict_pack_catches_iban() -> None:
    r = Redactor(packs=["strict"])
    out = r.redact("IBAN: DE89370400440532013000")
    assert "<REDACTED:iban>" in out


def test_phi_pack_catches_mrn_and_icd10_and_dob() -> None:
    r = Redactor(packs=["phi"])
    out = r.redact("Patient MRN:1234567 dx E11.9 dob 05/14/1982")
    assert "<REDACTED:mrn>" in out
    assert "<REDACTED:icd10>" in out
    assert "<REDACTED:dob>" in out


def test_finance_pack_catches_swift_bic() -> None:
    r = Redactor(packs=["finance"])
    out = r.redact("wire to DEUTDEFFXXX today")
    assert "<REDACTED:swift_bic>" in out


def test_secrets_pack_catches_aws_access_key() -> None:
    r = Redactor(packs=["secrets"])
    out = r.redact("key=AKIAIOSFODNN7EXAMPLE tail")
    assert "<REDACTED:aws_access_key>" in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_secrets_pack_catches_github_pat() -> None:
    r = Redactor(packs=["secrets"])
    out = r.redact("token ghp_" + "A" * 36 + " end")
    assert "<REDACTED:github_pat>" in out


def test_secrets_pack_catches_pem_block() -> None:
    r = Redactor(packs=["secrets"])
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBALZF1x\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = r.redact(f"key:\n{pem}\nend")
    assert "<REDACTED:pem_private_key>" in out
    assert "MIIBOgIBAAJBALZF1x" not in out


def test_unknown_pack_raises() -> None:
    with pytest.raises(ValueError, match="unknown pack"):
        Redactor(packs=["nonexistent"])


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------


def test_custom_rule_default_replacement_token() -> None:
    r = Redactor(
        packs=["standard"],
        rules=[Rule(name="internal_id", pattern=r"EMP-\d{6}")],
    )
    assert r.redact("email foo@bar.com, EMP-123456") == (
        "email <REDACTED:email>, <REDACTED:internal_id>"
    )


def test_custom_rule_with_explicit_replacement() -> None:
    r = Redactor(rules=[Rule(name="x", pattern=r"secret", replacement="***")])
    assert r.redact("a secret value") == "a *** value"


def test_custom_rule_accepts_compiled_pattern() -> None:
    pat = re.compile(r"[A-Z]{3}-\d{3}")
    r = Redactor(rules=[Rule(name="code", pattern=pat)])
    assert r.redact("ABC-123 and DEF-456") == "<REDACTED:code> and <REDACTED:code>"


# ---------------------------------------------------------------------------
# Pseudonymization
# ---------------------------------------------------------------------------


def test_pseudonymize_is_deterministic_with_same_salt() -> None:
    a = Redactor(packs=["standard"], pseudonymize=True, pseudonymize_salt="s3cret")
    b = Redactor(packs=["standard"], pseudonymize=True, pseudonymize_salt="s3cret")
    text = "email foo@bar.com"
    assert a.redact(text) == b.redact(text)
    assert a.redact(text).startswith("email <PSEUDO:")


def test_pseudonymize_changes_with_different_salt() -> None:
    a = Redactor(packs=["standard"], pseudonymize=True, pseudonymize_salt="one")
    b = Redactor(packs=["standard"], pseudonymize=True, pseudonymize_salt="two")
    assert a.redact("foo@bar.com") != b.redact("foo@bar.com")


def test_pseudonymize_without_salt_raises() -> None:
    with pytest.raises(ValueError, match="pseudonymize_salt"):
        Redactor(packs=["standard"], pseudonymize=True)


def test_per_rule_pseudonymize_opt_out() -> None:
    r = Redactor(
        packs=["standard"],
        pseudonymize=True,
        pseudonymize_salt="salty",
        rules=[Rule(name="emp", pattern=r"EMP-\d+", pseudonymize=False)],
    )
    out = r.redact("EMP-42 foo@bar.com")
    assert "<REDACTED:emp>" in out
    assert "<PSEUDO:" in out


# ---------------------------------------------------------------------------
# redact_span (field-path targeting)
# ---------------------------------------------------------------------------


def test_redact_span_only_touches_listed_fields() -> None:
    r = Redactor(packs=["standard"])
    span = {
        "id": "abc",
        "input": "ping me at foo@bar.com",
        "output": "SSN 111-22-3333",
        "untouched": "email ignored@example.com",
    }
    out = r.redact_span(span, fields=["input", "output"])
    assert "<REDACTED:email>" in out["input"]
    assert "<REDACTED:ssn>" in out["output"]
    # not listed => must be preserved verbatim
    assert out["untouched"] == "email ignored@example.com"
    assert out["id"] == "abc"


def test_redact_span_supports_dot_paths() -> None:
    r = Redactor(packs=["standard"])
    span = {
        "metadata": {"user": {"email": "foo@bar.com", "name": "alice"}},
    }
    out = r.redact_span(span, fields=["metadata.user.email"])
    assert out["metadata"]["user"]["email"] == "<REDACTED:email>"
    assert out["metadata"]["user"]["name"] == "alice"


def test_redact_span_missing_field_is_noop() -> None:
    r = Redactor(packs=["standard"])
    span = {"input": "foo@bar.com"}
    out = r.redact_span(span, fields=["input", "output", "metadata.nope"])
    assert out["input"] == "<REDACTED:email>"
    assert "output" not in out


def test_redact_span_handles_nested_structures() -> None:
    r = Redactor(packs=["standard"])
    span = {
        "input": ["hello", "contact foo@bar.com"],
        "output": {"msg": "SSN 111-22-3333", "n": 7},
    }
    out = r.redact_span(span, fields=["input", "output"])
    assert out["input"][1] == "contact <REDACTED:email>"
    assert out["input"][0] == "hello"
    assert out["output"]["msg"] == "SSN <REDACTED:ssn>"
    assert out["output"]["n"] == 7


# ---------------------------------------------------------------------------
# Module-level convenience and edge cases
# ---------------------------------------------------------------------------


def test_module_level_redact_helper() -> None:
    assert redact("ping foo@bar.com") == "ping <REDACTED:email>"


def test_redact_empty_string_is_noop() -> None:
    r = Redactor(packs=["standard"])
    assert r.redact("") == ""


def test_builtin_packs_are_exported() -> None:
    assert set(BUILTIN_PACKS.keys()) == {
        "standard",
        "strict",
        "phi",
        "finance",
        "secrets",
    }
