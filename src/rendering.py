from __future__ import annotations

import re
from typing import List, Tuple


_LATEX_COMMAND = re.compile(
    r"\\(?:frac|dfrac|tfrac|binom|sqrt|sum|prod|int|lim|mid|qquad|quad|cdot|times|"
    r"mu|sigma|theta|alpha|beta|gamma|delta|lambda|leq|geq|neq|approx|infty|"
    r"left|right|begin|end|mathrm|mathbf|text)\b"
)


def normalize_math_markdown(text: str) -> str:
    """Normalize common LLM LaTeX variants into Streamlit-friendly Markdown.

    Models are not always consistent about math delimiters. This function accepts
    the common display forms ``\\[...\\]`` and square-bracket-wrapped LaTeX and
    normalizes them to ``$$...$$``. Inline ``\\(...\\)`` is normalized to
    ``$...$``. Ordinary citations such as ``[S1]`` are left untouched.
    """
    if not text:
        return ""

    value = text.replace("\r\n", "\n")

    # Standard LaTeX display/inline delimiters sometimes emitted by LLMs.
    value = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
        value,
        flags=re.DOTALL,
    )
    value = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + m.group(1).strip() + "$",
        value,
        flags=re.DOTALL,
    )

    # Some models return display equations as: [ P(A|B)=\\frac{...}{...} ]
    # Convert only lines that contain a clear LaTeX command, so source citations
    # such as [S1] and normal prose in brackets are not modified.
    converted: List[str] = []
    for line in value.split("\n"):
        stripped = line.strip()
        if (
            len(stripped) >= 2
            and stripped.startswith("[")
            and stripped.endswith("]")
        ):
            body = stripped[1:-1].strip()
            if _LATEX_COMMAND.search(body):
                converted.extend(["$$", body, "$$"])
                continue
        converted.append(line)

    value = "\n".join(converted)

    # Keep display delimiters on their own lines for predictable splitting.
    value = re.sub(r"\s*\$\$\s*", "\n$$\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def split_display_math(text: str) -> List[Tuple[str, str]]:
    """Split normalized content into ('markdown'|'latex', content) blocks."""
    normalized = normalize_math_markdown(text)
    if not normalized:
        return []

    parts = normalized.split("$$")
    blocks: List[Tuple[str, str]] = []
    for index, part in enumerate(parts):
        content = part.strip()
        if not content:
            continue
        kind = "latex" if index % 2 == 1 else "markdown"
        blocks.append((kind, content))
    return blocks
