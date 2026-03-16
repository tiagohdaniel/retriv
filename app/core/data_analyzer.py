from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """Result of a deterministic data analysis operation."""
    summary: str                          # brief description for LLM context
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    result: object = None                 # computed value: scalar, list, or dict


class DataAnalyzerBase(ABC):
    """Interface for data analysis backends.

    Separates deterministic computation from the LLM layer.
    The LLM receives only the computed result to format — never raw data.

    analyze() accepts:
        question    — natural language question (used for intent inference)
        file_content — raw file bytes
        file_type   — "csv" | "xlsx" | "json"

    Returns AnalysisResult with the computation output and metadata.
    Swapping backends (pandas → DuckDB, SQL, etc.) requires only a new
    implementation and a config change in dependencies.py.
    """

    @abstractmethod
    def analyze(self, question: str, file_content: bytes, file_type: str) -> AnalysisResult: ...
