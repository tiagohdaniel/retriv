import re

# Each entry: (compiled pattern, replacement string).
# Applied in order — earlier rules take priority.
# Conservative: only maps well-known synonym pairs in legal/fiscal Portuguese.
# Rules must NOT inject values, numbers, or domain-specific answers.
_RULES: list[tuple[re.Pattern, str]] = [
    # --- Revenue vocabulary ---
    (re.compile(r'\breceita total\b', re.IGNORECASE), 'receita bruta'),
    (re.compile(r'\bfaturamento\b', re.IGNORECASE), 'receita bruta'),
    (re.compile(r'\brenda bruta\b', re.IGNORECASE), 'receita bruta'),
    (re.compile(r'\blimite de faturamento\b', re.IGNORECASE), 'limite de receita bruta'),
    (re.compile(r'\bteto de (?:receita|faturamento)\b', re.IGNORECASE), 'limite de receita bruta'),
    (re.compile(r'\bquanto (?:pode )?(?:ganhar|faturar|receber)\b', re.IGNORECASE), 'limite de receita bruta'),
    (re.compile(r'\bpor ano\b', re.IGNORECASE), 'no ano-calendário'),
    (re.compile(r'\banualmente\b', re.IGNORECASE), 'no ano-calendário'),

    # --- Legal entity types ---
    (re.compile(r'\bmicro\s*empresa\b', re.IGNORECASE), 'microempresa ME'),
    (re.compile(r'\bpequena empresa\b', re.IGNORECASE), 'empresa de pequeno porte EPP'),
    (re.compile(r'\bempresa de pequeno porte(?!\s+EPP)\b', re.IGNORECASE), 'empresa de pequeno porte EPP'),
]


class QueryNormalizer:
    """Deterministic vocabulary normalizer for legal/fiscal Portuguese queries.

    Bridges the gap between casual user language and the technical vocabulary
    used in legal/fiscal documents via a fixed synonym dictionary.

    No LLM, no randomness, no injected values or domain answers.
    If no rule matches, the original query is returned unchanged.
    """

    def normalize(self, query: str) -> tuple[str, bool]:
        """Return (normalized_query, was_changed)."""
        result = query
        for pattern, replacement in _RULES:
            result = pattern.sub(replacement, result)
        return result, result != query
