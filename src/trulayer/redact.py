"""
Client-side PII, secret, and PHI redaction.

Use :class:`Redactor` to scrub sensitive values from strings or structured
trace/span dictionaries before they leave your process. Redactors support:

* built-in entity packs (``standard``, ``strict``, ``phi``, ``finance``,
  ``secrets``)
* user-defined :class:`Rule` objects with arbitrary regular expressions
* pseudonymization (HMAC-SHA256 with a caller-provided salt) instead of
  simple replacement, controlled globally or per-rule
* field-path targeting via :meth:`Redactor.redact_span`

Example::

    from trulayer.redact import Redactor, Rule

    r = Redactor(
        packs=["standard"],
        rules=[Rule(name="internal_id", pattern=r"EMP-\\d{6}")],
    )
    r.redact("email foo@bar.com, EMP-123456")
    # -> "email <REDACTED:email>, <REDACTED:internal_id>"
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

__all__ = [
    "Rule",
    "Redactor",
    "redact",
    "BUILTIN_PACKS",
]


# ---------------------------------------------------------------------------
# Built-in pack definitions
# ---------------------------------------------------------------------------

# Each pack maps entity name -> (pattern, validator-or-None). The validator is
# an optional callable that receives the matched string and returns True if
# the match should be redacted. This lets us do Luhn checks on credit card
# candidates without redacting every 16-digit run.

_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
_PHONE = r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3}[\s.-]\d{3,4}(?!\d)"
_SSN = r"\b\d{3}-\d{2}-\d{4}\b"
_BEARER = r"Bearer\s+[A-Za-z0-9._\-]+"
_JWT = r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"

_IPV4 = r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
_CC_CANDIDATE = r"\b(?:\d[ -]?){13,19}\b"
_IBAN = r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"

_MRN = r"MRN[:\s]?\d{6,10}"
_ICD10 = r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b"
_DOB = (
    r"\b(?:"
    r"(?:0?[1-9]|1[0-2])[\/\-](?:0?[1-9]|[12]\d|3[01])[\/\-](?:19|20)\d{2}"
    r"|"
    r"(?:19|20)\d{2}[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:0?[1-9]|[12]\d|3[01])"
    r")\b"
)

_SWIFT = r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
_ROUTING = r"\b\d{9}\b"
_ACCOUNT = r"\b\d{8,17}\b"
_TICKER_AMOUNT = r"\$[A-Z]{1,5}\s*\$?\d+(?:,\d{3})*(?:\.\d+)?"

_AWS_KEY = r"\bAKIA[0-9A-Z]{16}\b"
_GITHUB_PAT = r"\bgh[pousr]_[A-Za-z0-9]{36}\b"
_PEM_BLOCK = (
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    r"[\s\S]+?"
    r"-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
)
# GCP service account keys are JSON documents carrying a private_key field.
_GCP_SA = r'"type"\s*:\s*"service_account"[\s\S]{0,200}?"private_key"\s*:\s*"[^"]+"'


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


BUILTIN_PACKS: dict[str, list[tuple[str, str, Any]]] = {
    # (entity_name, pattern, optional validator)
    "standard": [
        ("email", _EMAIL, None),
        ("ssn", _SSN, None),
        ("jwt", _JWT, None),
        ("bearer_token", _BEARER, None),
        ("phone", _PHONE, None),
    ],
    "strict": [
        ("email", _EMAIL, None),
        ("ssn", _SSN, None),
        ("jwt", _JWT, None),
        ("bearer_token", _BEARER, None),
        ("credit_card", _CC_CANDIDATE, _luhn_ok),
        ("iban", _IBAN, None),
        ("ipv4", _IPV4, None),
        ("phone", _PHONE, None),
    ],
    "phi": [
        ("mrn", _MRN, None),
        ("icd10", _ICD10, None),
        ("dob", _DOB, None),
    ],
    "finance": [
        ("swift_bic", _SWIFT, None),
        ("routing_number", _ROUTING, None),
        ("account_number", _ACCOUNT, None),
        ("ticker_amount", _TICKER_AMOUNT, None),
    ],
    "secrets": [
        ("aws_access_key", _AWS_KEY, None),
        ("github_pat", _GITHUB_PAT, None),
        ("pem_private_key", _PEM_BLOCK, None),
        ("gcp_service_account", _GCP_SA, None),
    ],
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """A user-defined redaction rule.

    :param name: short identifier used in the replacement token.
    :param pattern: regular expression string or pre-compiled pattern.
    :param replacement: token used when a match is redacted. When ``None``,
        the redactor emits ``<REDACTED:{name}>``.
    :param pseudonymize: when ``True``, matches are replaced with a HMAC
        pseudonym rather than a static token. Requires the redactor to be
        constructed with a ``pseudonymize_salt``.
    :param validator: optional callable that returns ``True`` when a match
        should actually be redacted. Used internally for Luhn-validated
        credit-card detection.
    """

    name: str
    pattern: str | re.Pattern[str]
    replacement: str | None = None
    pseudonymize: bool | None = None
    validator: Any = None

    _compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.pattern, re.Pattern):
            self._compiled = self.pattern
        else:
            self._compiled = re.compile(self.pattern)


class Redactor:
    """Compose packs and custom rules into a reusable scrubber.

    :param packs: names of built-in packs to enable.
    :param rules: additional custom rules applied after the packs.
    :param pseudonymize: when ``True``, all rules emit HMAC pseudonyms by
        default (individual rules can still opt out via ``Rule.pseudonymize=False``).
    :param pseudonymize_salt: secret used for HMAC-SHA256; required whenever
        any rule pseudonymizes.
    :param pseudonym_length: number of hex characters of the HMAC digest to
        retain in the emitted token. Defaults to 8.
    """

    def __init__(
        self,
        packs: Sequence[str] | None = None,
        rules: Sequence[Rule] | None = None,
        pseudonymize: bool = False,
        pseudonymize_salt: str | bytes | None = None,
        pseudonym_length: int = 8,
    ) -> None:
        if pseudonym_length < 4 or pseudonym_length > 64:
            raise ValueError("pseudonym_length must be between 4 and 64")

        self._pseudonymize_default = pseudonymize
        self._pseudonym_length = pseudonym_length
        if pseudonymize_salt is None:
            self._salt: bytes | None = None
        elif isinstance(pseudonymize_salt, str):
            self._salt = pseudonymize_salt.encode("utf-8")
        else:
            self._salt = bytes(pseudonymize_salt)

        self._rules: list[Rule] = []
        for pack in packs or []:
            if pack not in BUILTIN_PACKS:
                raise ValueError(
                    f"unknown pack '{pack}'; available: {sorted(BUILTIN_PACKS)}"
                )
            for name, pat, validator in BUILTIN_PACKS[pack]:
                self._rules.append(Rule(name=name, pattern=pat, validator=validator))
        for rule in rules or []:
            self._rules.append(rule)

        if self._needs_salt() and self._salt is None:
            raise ValueError(
                "pseudonymize=True requires pseudonymize_salt to be provided"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> str:
        """Return ``text`` with every configured rule applied."""
        if not isinstance(text, str) or not text:
            return text
        out = text
        for rule in self._rules:
            current_rule = rule  # capture for closure
            def _sub(m: re.Match[str], _r: Rule = current_rule) -> str:
                return self._replacement_for(_r, m.group(0))
            out = current_rule._compiled.sub(_sub, out)  # noqa: SLF001
        return out

    def redact_span(
        self,
        span: Mapping[str, Any],
        fields: Iterable[str] = ("input", "output", "metadata"),
    ) -> dict[str, Any]:
        """Return a shallow copy of ``span`` with the listed dot-path fields redacted.

        Fields may use dot notation (``metadata.user.email``) to target nested
        values. Missing fields are silently skipped. Every other key in the
        span is passed through unchanged.
        """
        result: dict[str, Any] = dict(span)
        for path in fields:
            self._apply_to_path(result, path.split("."))
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _needs_salt(self) -> bool:
        if self._pseudonymize_default:
            return True
        return any(r.pseudonymize for r in self._rules)

    def _replacement_for(self, rule: Rule, match: str) -> str:
        if rule.validator is not None and not rule.validator(match):
            return match
        use_pseudo = (
            rule.pseudonymize if rule.pseudonymize is not None else self._pseudonymize_default
        )
        if use_pseudo:
            if self._salt is None:
                # should never happen — constructor validates — but be defensive
                return f"<REDACTED:{rule.name}>"
            digest = hmac.new(self._salt, match.encode("utf-8"), sha256).hexdigest()
            return f"<PSEUDO:{digest[: self._pseudonym_length]}>"
        if rule.replacement is not None:
            return rule.replacement
        return f"<REDACTED:{rule.name}>"

    def _apply_to_path(self, root: Any, path: list[str]) -> None:
        if not path:
            return
        key = path[0]
        rest = path[1:]
        if isinstance(root, dict) and key in root:
            if not rest:
                root[key] = self._redact_value(root[key])
            else:
                self._apply_to_path(root[key], rest)
        elif isinstance(root, list):
            for item in root:
                self._apply_to_path(item, path)

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        return value


# ---------------------------------------------------------------------------
# Convenience module-level helper
# ---------------------------------------------------------------------------


def redact(
    text: str,
    packs: Sequence[str] = ("standard",),
    rules: Sequence[Rule] | None = None,
    pseudonymize: bool = False,
    pseudonymize_salt: str | bytes | None = None,
) -> str:
    """One-shot convenience wrapper around :class:`Redactor`.

    Constructs a redactor with the given packs/rules and applies it once to
    ``text``. For repeated use prefer instantiating a :class:`Redactor` so the
    regex compilation is amortized.
    """
    r = Redactor(
        packs=packs,
        rules=rules,
        pseudonymize=pseudonymize,
        pseudonymize_salt=pseudonymize_salt,
    )
    return r.redact(text)
