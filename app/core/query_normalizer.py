import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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


class FiscalQueryNormalizer(QueryNormalizerBase):
    """Vocabulary normalizer for Brazilian legal/fiscal documents.

    Bridges casual user language and technical terminology used in
    fiscal/legal documents (Simples Nacional, PGDAS-D, CLT, etc.).
    Rules are conservative: no values, numbers, or answers are injected.
    """

    # Each entry: (compiled pattern, replacement, rule_name)
    _RULES: list[tuple[re.Pattern, str, str]] = [
        # Revenue vocabulary
        (re.compile(r'\breceita total\b', re.IGNORECASE),              'receita bruta',            'receita_total'),
        (re.compile(r'\bfaturamento\b', re.IGNORECASE),                'receita bruta',            'faturamento'),
        (re.compile(r'\brenda bruta\b', re.IGNORECASE),                'receita bruta',            'renda_bruta'),
        (re.compile(r'\blimite de faturamento\b', re.IGNORECASE),      'limite de receita bruta',  'limite_faturamento'),
        (re.compile(r'\bteto de (?:receita|faturamento)\b', re.IGNORECASE), 'limite de receita bruta', 'teto_receita'),
        (re.compile(r'\bquanto (?:pode )?(?:ganhar|faturar|receber)\b', re.IGNORECASE), 'limite de receita bruta', 'quanto_ganhar'),
        (re.compile(r'\bpor ano\b', re.IGNORECASE),                    'no ano-calendário',        'por_ano'),
        (re.compile(r'\banualmente\b', re.IGNORECASE),                 'no ano-calendário',        'anualmente'),
        # Legal entity types
        (re.compile(r'\bmicro\s*empresa\b', re.IGNORECASE),            'microempresa ME',          'microempresa'),
        (re.compile(r'\bpequena empresa\b', re.IGNORECASE),            'empresa de pequeno porte EPP', 'pequena_empresa'),
        (re.compile(r'\bempresa de pequeno porte(?!\s+EPP)\b', re.IGNORECASE), 'empresa de pequeno porte EPP', 'epp_acronym'),
    ]

    def normalize(self, query: str) -> NormalizedQueryResult:
        result = query
        applied: list[str] = []
        for pattern, replacement, name in self._RULES:
            new = pattern.sub(replacement, result)
            if new != result:
                applied.append(name)
            result = new
        return NormalizedQueryResult(original=query, normalized=result, rules_applied=applied)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, QueryNormalizerBase] = {
    "fiscal": FiscalQueryNormalizer(),
    "none":   NullQueryNormalizer(),
}


def get_normalizer(domain: str = "fiscal") -> QueryNormalizerBase:
    """Return the normalizer for the given domain.

    Falls back to NullQueryNormalizer for unknown domains so the
    pipeline never breaks when a new domain is configured but not
    yet implemented.

    To add a new domain:
        1. Create a class MyDomainQueryNormalizer(QueryNormalizerBase)
        2. Add it to _REGISTRY: _REGISTRY["my_domain"] = MyDomainQueryNormalizer()
        3. Set QUERY_NORMALIZER_DOMAIN=my_domain in the environment
    """
    return _REGISTRY.get(domain.lower(), NullQueryNormalizer())
