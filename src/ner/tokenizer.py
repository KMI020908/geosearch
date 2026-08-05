"""Word splitter shared by the GLiNER fine-tuning export and GLiNER inference.

GLiNER does not predict character offsets: it enumerates *token* spans and
classifies each one, so a span it cannot express as a whole number of tokens is
a span it can neither be trained on nor predict. Which tokens exist is decided by
the model's word splitter (``data_processor.words_splitter``), and GLiNER's
default one is whitespace-based::

    re.compile(r"\\w+(?:[-_]\\w+)*|\\S")

``\\w`` is unicode-aware, so a run of Han characters is one token. Chinese text
carries no spaces, so ``莫斯科新闻`` ("Moscow news") is a *single* token and the
city span ``莫斯科`` is structurally unrepresentable. Measured on
``data/query_dataset.parquet``: **all 56 zh spans** fail to align to whitespace
tokens, versus 0 for ru/en/tr.

:class:`CjkAwareSplitter` fixes exactly that one thing — every CJK ideograph
becomes its own token, everything else keeps GLiNER's own rule — which brings the
alignment failures to 0/410. It costs longer token sequences for zh (one token
per character, mean 8.7 / max 93 tokens per query in the current dataset, well
under GLiNER's ``max_len``), and it shifts zh tokenisation away from what the
pretrained checkpoint saw, which is a cost fine-tuning pays back and zero-shot
inference does not.

**Train/serve parity**: a model fine-tuned on data tokenised this way must be
served with this splitter too, i.e. after loading::

    model.data_processor.words_splitter = CjkAwareSplitter()

Nothing enforces that automatically — the splitter is not part of the saved
checkpoint (``GLiNERConfig.words_splitter_type`` only names the built-in kinds).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# Han ideographs: BMP blocks (incl. Extension A and the compatibility block) plus
# the supplementary planes, where rare characters used in place names live
# (e.g. 𠯫, U+20BEB, which appears in this dataset).
_CJK_RANGES = (
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "\U00020000-\U0003134f"  # Extensions B-G (supplementary planes)
)

# The CJK alternative comes *first* so that `\w+` — which also matches Han
# characters — cannot swallow a run of them into one token.
_TOKEN_PATTERN = re.compile(rf"[{_CJK_RANGES}]|\w+(?:[-_]\w+)*|\S")

# One token as GLiNER's splitter protocol yields it: (text, start, end),
# with `end` exclusive (Python slice convention), like `re.Match.end()`.
Token = tuple[str, int, int]


class CjkAwareSplitter:
    """GLiNER's whitespace splitter, with CJK ideographs split per character.

    Callable with the same ``(token, start, end)`` generator protocol as
    ``gliner.data_processing.tokenizer.WhitespaceTokenSplitter``, so it can be
    assigned straight onto ``model.data_processor.words_splitter``.
    """

    def __call__(self, text: str) -> Iterator[Token]:
        """Yield ``(token, start, end)`` for every token in ``text``."""
        for match in _TOKEN_PATTERN.finditer(text):
            yield match.group(), match.start(), match.end()


def tokenize(text: str) -> list[Token]:
    """Tokenise ``text`` into a list of ``(token, start, end)``."""
    return list(CjkAwareSplitter()(text))


def align_span(tokens: list[Token], start: int, end: int) -> tuple[int, int] | None:
    """Map a character span onto inclusive token indices, or ``None``.

    ``None`` means the character span does not coincide with token boundaries
    (it starts or ends inside a token), so GLiNER could never predict it. The
    check is deliberately strict rather than snapping outwards to the enclosing
    tokens: a widened span teaches the model to tag ``莫斯科新闻`` ("Moscow news")
    as a city, which is a labelling error, not a rounding error.
    """
    first = next((i for i, (_, s, _) in enumerate(tokens) if s == start), None)
    last = next((i for i, (_, _, e) in enumerate(tokens) if e == end), None)
    if first is None or last is None or last < first:
        return None
    return first, last
