from src.rendering import normalize_math_markdown, split_display_math


def test_bracket_wrapped_latex_becomes_display_math():
    raw = r"[ P(A_i\mid G)=\frac{P(G\mid A_i)P(A_i)}{P(G)} ]"
    normalized = normalize_math_markdown(raw)
    assert normalized.startswith("$$")
    assert r"\frac" in normalized
    assert normalized.endswith("$$")


def test_standard_latex_delimiters_are_normalized():
    raw = r"Before \[ x=\frac{1}{2} \] after \(p=0.5\)."
    normalized = normalize_math_markdown(raw)
    assert "$$" in normalized
    assert "$p=0.5$" in normalized


def test_citations_are_not_changed():
    raw = "The value is supported by [S1] and [S2]."
    assert normalize_math_markdown(raw) == raw


def test_display_math_splits_into_renderable_blocks():
    raw = r"Explanation\n\n\[ P(X=x)=\binom{n}{x}p^x(1-p)^{n-x} \]\n\nDone."
    blocks = split_display_math(raw)
    assert [kind for kind, _ in blocks] == ["markdown", "latex", "markdown"]
