"""The structured response contract for the query generator.

DeepSeek's JSON mode only supports ``response_format={"type": "json_object"}`` —
there is no server-side schema enforcement — so the schema lives here and is
applied on our side: anything that does not parse into :class:`GeneratedQuery`
is a failed request, retried on the next warm-start run.

The model returns ``text`` and ``label`` for each entity; ``start`` / ``end`` are
filled by :func:`src.dataset.generate.locate_spans` from the query itself, never
by the model — offsets are exactly the kind of thing an LLM gets subtly wrong,
and ``str.find`` cannot.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config import settings


class Entity(BaseModel):
    """One NER-shaped span of the generated query.

    ``text`` is the surface form as it appears in the query (inflected, not
    lemmatised); ``label`` is one of ``settings.ner_labels``, i.e. the same
    vocabulary GLiNER emits at serve time.
    """

    text: str
    label: str
    start: int = -1
    end: int = -1

    @field_validator("text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("entity text is empty")
        return text

    @field_validator("label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        label = value.strip().upper()
        if label not in settings.ner_labels:
            raise ValueError(
                f"unknown label {value!r}; expected one of {settings.ner_labels}"
            )
        return label


class GeneratedQuery(BaseModel):
    """One generation result: the query, its entities, and self-reported confidence.

    ``confidence`` is the model's own estimate that it followed every rule of its
    prompt (right language, every requested name present, spans copied exactly).
    It is a soft signal — the deterministic checks in
    :func:`src.dataset.generate.validate_generation` are what actually gate a row —
    kept as a column so a threshold can be picked once its distribution is known.
    """

    query: str
    entities: list[Entity] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query is empty")
        return query


def parse_response(content: str) -> GeneratedQuery:
    """Parse one raw completion into a :class:`GeneratedQuery`.

    JSON mode returns bare JSON, but a stray ```json fence costs a whole request
    to reject, so it is stripped before parsing.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the opening ```json line
        text = text.rsplit("```", 1)[0].strip()  # drop the closing fence
    return GeneratedQuery.model_validate_json(text)
