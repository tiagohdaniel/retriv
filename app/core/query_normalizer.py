import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypedDict


@dataclass
class NormalizedQueryResult:
    original: str
    normalized: str
    rules_applied: list[str] = field(default_factory=list)

    @property
    def was_changed(self) -> bool:
        return self.original != self.normalized


class QueryNormalizerBase(ABC):
    """Interface for query vocabulary normalizers.

    Each domain implements its own normalizer. The retrieval layer
    depends only on this interface — never on a concrete implementation.
    """

    @abstractmethod
    def normalize(self, query: str) -> NormalizedQueryResult: ...


class NullQueryNormalizer(QueryNormalizerBase):
    """No-op — used when no domain normalization is needed."""

    def normalize(self, query: str) -> NormalizedQueryResult:
        return NormalizedQueryResult(original=query, normalized=query)


# ---------------------------------------------------------------------------
# Generic rule-based engine
# ---------------------------------------------------------------------------

class RuleSpec(TypedDict):
    pattern: str
    replacement: str
    name: str


class RuleBasedQueryNormalizer(QueryNormalizerBase):
    """Generic engine: domain = set of rules passed at construction time.

    Each rule maps an informal query pattern to the technical vocabulary
    used in indexed documents. Rules are applied in order — a query can
    be modified by more than one rule in a single pass.

    Rules should:
    - Use word boundaries (\\b) to avoid false positives on technical queries
      that already work (e.g. "faturamento" → "receita bruta" must not fire
      on "faturamento bruto de exportação" if that phrase already retrieves)
    - Only substitute vocabulary, never inject answers or values
    - Have descriptive names — they appear in logs and Langfuse traces
    """

    def __init__(self, rules: list[RuleSpec]):
        self._rules: list[tuple[re.Pattern, str, str]] = [
            (re.compile(r["pattern"], re.IGNORECASE), r["replacement"], r["name"])
            for r in rules
        ]

    def normalize(self, query: str) -> NormalizedQueryResult:
        result = query
        applied: list[str] = []
        for pattern, replacement, name in self._rules:
            new = pattern.sub(replacement, result)
            if new != result:
                applied.append(name)
            result = new
        return NormalizedQueryResult(original=query, normalized=result, rules_applied=applied)


# ---------------------------------------------------------------------------
# Domain rule sets  (data, not classes)
# Keep focused: max ~10 high-impact rules per domain.
# Add a rule only after confirming the informal term causes a real retrieval
# miss — a rule that fires incorrectly degrades queries that already work.
# ---------------------------------------------------------------------------

_FISCAL_RULES: list[RuleSpec] = [
    # Revenue vocabulary
    {"pattern": r"\breceita total\b",                                   "replacement": "receita bruta",              "name": "receita_total"},
    {"pattern": r"\bfaturamento\b",                                     "replacement": "receita bruta",              "name": "faturamento"},
    {"pattern": r"\brenda bruta\b",                                     "replacement": "receita bruta",              "name": "renda_bruta"},
    {"pattern": r"\blimite de faturamento\b",                           "replacement": "limite de receita bruta",    "name": "limite_faturamento"},
    {"pattern": r"\bteto de (?:receita|faturamento)\b",                 "replacement": "limite de receita bruta",    "name": "teto_receita"},
    {"pattern": r"\bquanto (?:pode )?(?:ganhar|faturar|receber)\b",     "replacement": "limite de receita bruta",    "name": "quanto_ganhar"},
    {"pattern": r"\bpor ano\b",                                         "replacement": "no ano-calendário",          "name": "por_ano"},
    {"pattern": r"\banualmente\b",                                      "replacement": "no ano-calendário",          "name": "anualmente"},
    # Legal entity types
    {"pattern": r"\bmicro\s*empresa\b",                                 "replacement": "microempresa ME",            "name": "microempresa"},
    {"pattern": r"\bpequena empresa\b",                                 "replacement": "empresa de pequeno porte EPP", "name": "pequena_empresa"},
    {"pattern": r"\bempresa de pequeno porte(?!\s+EPP)\b",              "replacement": "empresa de pequeno porte EPP", "name": "epp_acronym"},
]

_ACCOUNTING_RULES: list[RuleSpec] = [
    # IMPORTANT: multi-word (specific) rules must come before single-word
    # (general) rules to avoid partial substitutions that corrupt the phrase.
    # e.g. "teto de faturamento" must be caught before the bare "faturamento"
    # rule fires and turns it into "teto de receita bruta" (breaking the match).

    # Revenue ceiling — specific multi-word first
    {"pattern": r"\bteto de (?:receita|faturamento)\b",                 "replacement": "limite de receita bruta",    "name": "teto_receita"},
    {"pattern": r"\blimite de faturamento\b",                           "replacement": "limite de receita bruta",    "name": "limite_faturamento"},

    # "quanto/posso ganhar/faturar" — covers "quanto posso ganhar" AND
    # "valor máximo que eu posso ganhar" (Q06 case confirmed in benchmarks)
    {"pattern": r"\bquanto (?:pode |posso )?(?:ganhar|faturar|receber)\b", "replacement": "limite de receita bruta", "name": "quanto_ganhar"},
    {"pattern": r"\bposso (?:ganhar|faturar|receber)\b",                "replacement": "limite de receita bruta",    "name": "posso_ganhar"},

    # General single-word substitution — after specific patterns above
    {"pattern": r"\bfaturamento\b",                                     "replacement": "receita bruta",              "name": "faturamento"},

    # Export
    {"pattern": r"\bvender(?: p(?:a)?ra)? fora\b",                     "replacement": "exportação receita",         "name": "vender_fora"},

    # Entity types
    {"pattern": r"\bmicro\s*empresa\b",                                 "replacement": "microempresa ME",            "name": "microempresa"},
    {"pattern": r"\bpequena empresa\b",                                 "replacement": "empresa de pequeno porte EPP", "name": "pequena_empresa"},
]


# ---------------------------------------------------------------------------
# Backward-compatible alias: FiscalQueryNormalizer stays in the public API
# so existing code (imports, tests) keeps working without changes.
# ---------------------------------------------------------------------------

class FiscalQueryNormalizer(RuleBasedQueryNormalizer):
    """Fiscal/legal vocabulary normalizer (Brazilian Simples Nacional docs).

    Kept for backward compatibility. Internally delegates to
    RuleBasedQueryNormalizer with _FISCAL_RULES.
    """

    def __init__(self):
        super().__init__(_FISCAL_RULES)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, QueryNormalizerBase] = {
    "fiscal":      FiscalQueryNormalizer(),
    "accounting":  RuleBasedQueryNormalizer(_ACCOUNTING_RULES),
    "none":        NullQueryNormalizer(),
}


def get_normalizer(domain: str = "fiscal") -> QueryNormalizerBase:
    """Return the normalizer for the given domain.

    Falls back to NullQueryNormalizer for unknown domains so the
    pipeline never breaks when a new domain is configured but not
    yet implemented.

    To add a new domain:
        1. Define _MY_DOMAIN_RULES: list[RuleSpec] = [...]
        2. Add to registry: _REGISTRY["my_domain"] = RuleBasedQueryNormalizer(_MY_DOMAIN_RULES)
        3. Set QUERY_NORMALIZER_DOMAIN=my_domain in the environment
    """
    return _REGISTRY.get(domain.lower(), NullQueryNormalizer())
