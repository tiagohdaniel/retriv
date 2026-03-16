"""Pandas adapter for DataAnalyzerBase.

Responsible for:
1. Parsing CSV / XLSX / JSON bytes into a DataFrame
2. Inferring the operation from the question (heuristics, no LLM)
3. Executing the computation deterministically
4. Returning an AnalysisResult for the service layer to pass to the LLM

The LLM never sees raw rows — only the computed result and its summary.
This prevents hallucinated aggregations and bounds token usage to O(1).
"""
import io
import re

import pandas as pd

from app.core.data_analyzer import DataAnalyzerBase, AnalysisResult


# ---------------------------------------------------------------------------
# Intent patterns (order matters — more specific first)
# ---------------------------------------------------------------------------

_OP_PATTERNS: list[tuple[str, list[str]]] = [
    ("mean",    [r"\bm[eé]dia\b", r"\bm[eé]dio\b", r"\baverage\b", r"\bmean\b"]),
    ("max",     [r"\bm[aá]ximo\b", r"\bmaior\b", r"\bhighest\b", r"\bmax\b"]),
    ("min",     [r"\bm[ií]nimo\b", r"\bmenor\b", r"\blowest\b", r"\bmin\b"]),
    ("count",   [r"\bquantos\b", r"\bquantas\b", r"\bcount\b", r"\bcontar\b", r"\bnúmero de\b", r"\bnumero de\b", r"\bhow many\b"]),
    ("sum",     [r"\btotal\b", r"\bsoma\b", r"\bsom[ae]\b", r"\bsum\b", r"\bsomando\b", r"\bsomados\b"]),
    ("list",    [r"\blist[ae]?\b", r"\bmostr[ae]\b", r"\bexib[ae]\b", r"\btop\s+\d+\b"]),
]


def _infer_op(question: str) -> str:
    q = question.lower()
    for op, patterns in _OP_PATTERNS:
        for pat in patterns:
            if re.search(pat, q):
                return op
    return "describe"


def _find_column(question: str, columns: list[str]) -> str | None:
    """Return the first column name found (case-insensitive) in the question."""
    q = question.lower()
    for col in columns:
        if col.lower() in q:
            return col
    return None


def _find_group_column(question: str, columns: list[str]) -> str | None:
    """Return the column after 'por' / 'by' / 'group by' in the question, if it exists."""
    q = question.lower()
    match = re.search(r"\b(?:por|by|group by|agrupado por)\s+(\w[\w\s]*)", q)
    if not match:
        return None
    fragment = match.group(1).strip()
    for col in columns:
        if col.lower() in fragment:
            return col
    return None


def _top_n(question: str, default: int = 10) -> int:
    match = re.search(r"(?:\btop\b|\bprimeiros?\b|\bprimeiras?\b|\bfirst\b)\s+(\d+)", question.lower())
    return int(match.group(1)) if match else default


class PandasAnalyzer(DataAnalyzerBase):
    """Analyzes tabular data using pandas."""

    def __init__(self, max_rows: int = 100_000) -> None:
        self.max_rows = max_rows

    def analyze(self, question: str, file_content: bytes, file_type: str) -> AnalysisResult:
        df = self._load(file_content, file_type)

        if len(df) > self.max_rows:
            raise ValueError(
                f"Dataset has {len(df):,} rows — limit is {self.max_rows:,}. "
                "Use a filtered export or a smaller sample."
            )

        cols = list(df.columns)
        op = _infer_op(question)
        target_col = _find_column(question, cols)
        group_col = _find_group_column(question, cols)

        result, summary = self._execute(df, op, target_col, group_col, question)

        return AnalysisResult(
            summary=summary,
            columns=cols,
            row_count=len(df),
            result=result,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, content: bytes, file_type: str) -> pd.DataFrame:
        buf = io.BytesIO(content)
        if file_type == "csv":
            return pd.read_csv(buf)
        if file_type == "xlsx":
            return pd.read_excel(buf, engine="openpyxl")
        if file_type == "json":
            return pd.read_json(buf)
        raise ValueError(f"Unsupported file type: '{file_type}'. Supported: csv, xlsx, json.")

    def _execute(
        self,
        df: pd.DataFrame,
        op: str,
        target_col: str | None,
        group_col: str | None,
        question: str,
    ) -> tuple[object, str]:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Fallback to first numeric column when no match found
        if target_col is None and numeric_cols:
            target_col = numeric_cols[0]

        if op == "count":
            if group_col:
                result = df.groupby(group_col).size().sort_values(ascending=False).to_dict()
                summary = f"Count per '{group_col}'"
            else:
                result = len(df)
                summary = f"Total row count: {result:,}"
            return result, summary

        if op == "list":
            n = _top_n(question)
            result = df.head(n).to_dict(orient="records")
            summary = f"First {min(n, len(df))} rows of the dataset"
            return result, summary

        if op == "describe":
            if target_col and target_col in numeric_cols:
                stats = df[target_col].describe()
                result = {k: round(v, 4) for k, v in stats.items()}
                summary = f"Descriptive statistics for column '{target_col}'"
            else:
                result = {col: round(df[col].describe().to_dict().get("mean", 0), 4)
                          for col in numeric_cols[:5]}
                summary = f"Summary statistics for numeric columns"
            return result, summary

        # Aggregation ops: sum, mean, max, min
        if target_col is None:
            raise ValueError(
                "Could not identify the target column from the question. "
                "Please mention the column name explicitly."
            )

        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found. Available columns: {list(df.columns)}")

        agg_func = {"sum": "sum", "mean": "mean", "max": "max", "min": "min"}[op]

        if group_col:
            result = (
                df.groupby(group_col)[target_col]
                .agg(agg_func)
                .sort_values(ascending=(op == "min"))
                .round(4)
                .to_dict()
            )
            summary = f"{op.capitalize()} of '{target_col}' grouped by '{group_col}'"
        else:
            raw = getattr(df[target_col], agg_func)()
            result = round(float(raw), 4)
            summary = f"{op.capitalize()} of '{target_col}': {result}"

        return result, summary
