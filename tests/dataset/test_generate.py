"""Unit tests for the query generator's structured-output contract — no network.

DeepSeek's json mode has no server-side schema enforcement, so everything that
keeps a generated row honest lives on our side: parsing
(:mod:`src.dataset.schemas`), span location and the per-type entity checks
(:mod:`src.dataset.generate`). These tests cover exactly that layer, plus the one
prompt property the API itself requires (the literal word "json").
"""

import json
from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import ValidationError

from src.config import DatasetConfig
from src.dataset.generate import (
    UsageStats,
    _checkpoint_row,
    _expected_city_count,
    checkpoint_to_parquet,
    generate_validated,
    group_order,
    load_done_ids,
    locate_spans,
    validate_generation,
)
from src.dataset.prompts import PROMPT_VERSION, PROMPTS, build_messages
from src.dataset.sampling import KINDS
from src.dataset.schemas import Entity, GeneratedQuery, parse_response


def _gen(query: str, spans: list[tuple[str, str]], confidence: float = 1.0):
    return GeneratedQuery(
        query=query,
        entities=[Entity(text=text, label=label) for text, label in spans],
        confidence=confidence,
    )


def _row(name: str, sample_source: str) -> dict:
    return {"name": name, "sample_source": sample_source}


# ---------------------------------------------------------------------------
# locate_spans
# ---------------------------------------------------------------------------


def test_locate_spans_fills_offsets_from_the_query():
    query = "отели в Москве недорого"
    located = locate_spans(query, [Entity(text="Москве", label="CITY")])

    assert located is not None
    assert (located[0].start, located[0].end) == (8, 14)
    assert query[located[0].start : located[0].end] == "Москве"


def test_locate_spans_walks_forward_for_a_repeated_name():
    """Two spans of the same text must get two distinct offsets, not both the first."""
    query = "Москва Москва разница"
    located = locate_spans(
        query,
        [Entity(text="Москва", label="CITY"), Entity(text="Москва", label="CITY")],
    )

    assert located is not None
    assert [(e.start, e.end) for e in located] == [(0, 6), (7, 13)]


def test_locate_spans_tolerates_spans_listed_out_of_order():
    query = "из Казани в Москву"
    located = locate_spans(
        query,
        [Entity(text="Москву", label="CITY"), Entity(text="Казани", label="CITY")],
    )

    assert located is not None
    assert [e.start for e in located] == [12, 3]


def test_locate_spans_rejects_a_lemmatised_span():
    """The check that catches a model that re-spelled instead of copying."""
    assert locate_spans("отели в Москве", [Entity(text="Москва", label="CITY")]) is None


# ---------------------------------------------------------------------------
# validate_generation
# ---------------------------------------------------------------------------


def test_one_city_happy_path():
    _, reason = validate_generation(
        _row("Москва", "one_city"),
        PROMPTS["one_city"],
        _gen("что посмотреть в Москве за один день", [("Москве", "CITY")]),
    )
    assert reason is None


def test_one_city_rejects_a_stray_country():
    _, reason = validate_generation(
        _row("Москва", "one_city"),
        PROMPTS["one_city"],
        _gen(
            "что посмотреть в Москве, Россия",
            [("Москве", "CITY"), ("Россия", "COUNTRY")],
        ),
    )
    assert reason is not None
    assert "COUNTRY" in reason


def test_one_city_rejects_a_bare_name():
    _, reason = validate_generation(
        _row("Москва", "one_city"),
        PROMPTS["one_city"],
        _gen("Москва", [("Москва", "CITY")]),
    )
    assert reason == "query is the bare name"


def test_multi_city_requires_one_span_per_requested_name():
    row = _row("Москва, Казань, Сочи", "multi_city")
    assert _expected_city_count(row, PROMPTS["multi_city"]) == 3

    _, reason = validate_generation(
        row,
        PROMPTS["multi_city"],
        _gen("рейсы Москва Казань", [("Москва", "CITY"), ("Казань", "CITY")]),
    )
    assert reason is not None
    assert "2" in reason and "CITY" in reason


def test_multi_city_happy_path():
    _, reason = validate_generation(
        _row("Москва, Казань", "multi_city"),
        PROMPTS["multi_city"],
        _gen(
            "как добраться из Москвы в Казань", [("Москвы", "CITY"), ("Казань", "CITY")]
        ),
    )
    assert reason is None


@pytest.mark.parametrize("admin1_label", ["REGION", "STATE"])
def test_city_admin1_accepts_either_admin1_label(admin1_label):
    """REGION and STATE both mean admin1 (LABEL_BUCKETS), so both satisfy it."""
    _, reason = validate_generation(
        _row("Ростов", "city_admin1"),
        PROMPTS["city_admin1"],
        _gen(
            "погода в Ростове, Ростовская область",
            [("Ростове", "CITY"), ("Ростовская область", admin1_label)],
        ),
    )
    assert reason is None


def test_city_admin1_rejects_a_missing_region():
    _, reason = validate_generation(
        _row("Ростов", "city_admin1"),
        PROMPTS["city_admin1"],
        _gen("погода в Ростове", [("Ростове", "CITY")]),
    )
    assert reason is not None
    assert "admin1" in reason


def test_city_country_happy_path():
    _, reason = validate_generation(
        _row("Moscow", "city_country"),
        PROMPTS["city_country"],
        _gen(
            "flights to Moscow in the United States",
            [("Moscow", "CITY"), ("United States", "COUNTRY")],
        ),
    )
    assert reason is None


def test_city_admin1_country_requires_all_three():
    _, reason = validate_generation(
        _row("Moscow", "city_admin1_country"),
        PROMPTS["city_admin1_country"],
        _gen(
            "moscow idaho united states population",
            [("moscow", "CITY"), ("idaho", "STATE"), ("united states", "COUNTRY")],
        ),
    )
    assert reason is None


def test_validation_reports_an_unlocatable_span_first():
    _, reason = validate_generation(
        _row("Москва", "one_city"),
        PROMPTS["one_city"],
        _gen("отели в Москве", [("Москва", "CITY")]),
    )
    assert reason is not None
    assert "not found verbatim" in reason


def test_validation_returns_the_generation_with_offsets_filled():
    generation, reason = validate_generation(
        _row("Москва", "one_city"),
        PROMPTS["one_city"],
        _gen("отели в Москве", [("Москве", "CITY")]),
    )
    assert reason is None
    assert (generation.entities[0].start, generation.entities[0].end) == (8, 14)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def test_parse_response_reads_bare_json():
    generation = parse_response(
        '{"query": "отели в Москве", "entities": '
        '[{"text": "Москве", "label": "CITY"}], "confidence": 0.9}'
    )
    assert generation.query == "отели в Москве"
    assert generation.entities[0].label == "CITY"
    assert generation.confidence == 0.9


def test_parse_response_strips_a_markdown_fence():
    generation = parse_response(
        '```json\n{"query": "hotels in Moscow", "entities": [], "confidence": 1.0}\n```'
    )
    assert generation.query == "hotels in Moscow"


def test_parse_response_lowercases_nothing_but_normalises_label_case():
    generation = parse_response(
        '{"query": "hotels in Moscow", "entities": '
        '[{"text": "Moscow", "label": "city"}], "confidence": 1.0}'
    )
    assert generation.entities[0].label == "CITY"


def test_parse_response_rejects_an_unknown_label():
    with pytest.raises(ValidationError):
        parse_response(
            '{"query": "q", "entities": [{"text": "Neva", "label": "RIVER"}], '
            '"confidence": 1.0}'
        )


def test_parse_response_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        parse_response('{"query": "q", "entities": [], "confidence": 1.5}')


def test_parse_response_rejects_a_missing_confidence():
    """The score is the point of the contract, so it is required, not defaulted."""
    with pytest.raises(ValidationError):
        parse_response('{"query": "q", "entities": []}')


def test_parse_response_rejects_an_empty_query():
    with pytest.raises(ValidationError):
        parse_response('{"query": "   ", "entities": [], "confidence": 1.0}')


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _plan_row(sample_source: str) -> dict:
    return {
        "sample_source": sample_source,
        "isolanguage": "ru",
        "name": "Ростов",
        "admin1_name": "Rostov Oblast",
        "country_name": "Россия",
        "style": "search",
        "topic": "hotels",
    }


def test_every_query_type_has_its_own_system_prompt():
    prompts = {src: build_messages(_plan_row(src))[0]["content"] for src in PROMPTS}
    assert len(set(prompts.values())) == len(PROMPTS)


@pytest.mark.parametrize("sample_source", sorted(PROMPTS))
def test_every_system_prompt_says_json(sample_source):
    """DeepSeek's json_object mode requires the literal word in the prompt."""
    system = build_messages(_plan_row(sample_source))[0]["content"]
    assert "json" in system


def test_one_city_user_message_carries_no_region_or_country():
    user = build_messages(_plan_row("one_city"))[1]["content"]
    assert "region:" not in user
    assert "country:" not in user
    assert "name: Ростов" in user


def test_city_admin1_country_user_message_carries_both():
    user = build_messages(_plan_row("city_admin1_country"))[1]["content"]
    assert "region: Rostov Oblast" in user
    assert "country: Россия" in user


def test_unknown_sample_source_is_a_loud_error():
    with pytest.raises(KeyError, match="no prompt for sample_source"):
        build_messages(_plan_row("city_admin2"))


def test_group_order_follows_prompt_declaration_order():
    plan = pl.DataFrame({"sample_source": ["city_country", "one_city", "one_city"]})
    assert group_order(plan) == ["one_city", "city_country"]


def test_group_order_is_unaffected_by_the_pool_column():
    """The homonym/unique split must not add generation groups.

    Each group pays one cold request to warm its prefix cache, and a source that
    is not a ``PROMPTS`` key would be dropped from generation entirely — which is
    why the pool rides along as a column instead of as its own ``sample_source``.
    """
    plan = pl.DataFrame(
        {
            "sample_source": [spec.sample_source for spec in KINDS.values()],
            "pool": [spec.pool for spec in KINDS.values()],
        }
    )

    assert group_order(plan) == [s for s in PROMPTS if s in set(plan["sample_source"])]
    assert len(group_order(plan)) <= len(PROMPTS)


# ---------------------------------------------------------------------------
# Checkpoint -> Parquet
# ---------------------------------------------------------------------------


def _span(text: str, label: str, start: int, end: int) -> dict:
    return {"text": text, "label": label, "start": start, "end": end}


def _plan_fields(request_id: int, sample_source: str) -> dict:
    return {
        "request_id": request_id,
        "geonameid": [1],
        "name": "X",
        "isolanguage": "ru",
        "style": "search",
        "topic": "hotels",
        "sample_source": sample_source,
        "pool": "all",
        "admin1_name": "",
        "country_name": "",
    }


def _current_row(
    request_id: int,
    query: str,
    entities: list[dict],
    *,
    valid: bool = True,
    version: int = PROMPT_VERSION,
    sample_source: str = "one_city",
) -> dict:
    return _plan_fields(request_id, sample_source) | {
        "query": query,
        "entities": entities,
        "confidence": 0.9,
        "prompt_version": version,
        "valid": valid,
        "invalid_reason": "" if valid else "expected 0 COUNTRY span(s), got 1",
        "attempts": 1,
    }


def _legacy_row(request_id: int, query: str) -> dict:
    """A row in the pre-structured-output shape: no entities, no prompt_version."""
    return _plan_fields(request_id, "one_city") | {"query": query}


def _write(path, rows: list[dict]) -> str:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_load_done_ids_ignores_legacy_and_older_prompt_versions(tmp_path):
    """A prompt change must invalidate its rows, not silently mix versions."""
    checkpoint = _write(
        tmp_path / "ck.jsonl",
        [
            _legacy_row(1, "stale row from the old contract"),
            _current_row(2, "отели в Москве", [_span("Москве", "CITY", 8, 14)]),
            _current_row(3, "old prompt", [_span("old", "CITY", 0, 3)], version=1),
        ],
    )
    assert load_done_ids(checkpoint) == {2}


def test_load_done_ids_on_a_missing_checkpoint():
    assert load_done_ids("/nonexistent/ck.jsonl") == set()


def test_load_done_ids_skips_a_truncated_line(tmp_path):
    """A process killed mid-write must not make every future run unstartable."""
    path = tmp_path / "ck.jsonl"
    good = json.dumps(
        _current_row(2, "отели в Москве", [_span("Москве", "CITY", 8, 14)]),
        ensure_ascii=False,
    )
    path.write_text(f'{good}\n{{"request_id": 3, "prom', encoding="utf-8")

    assert load_done_ids(str(path)) == {2}


def test_load_done_ids_skips_a_row_without_a_request_id(tmp_path):
    checkpoint = _write(
        tmp_path / "ck.jsonl",
        [
            {"prompt_version": PROMPT_VERSION, "query": "no id here"},
            _current_row(7, "отели в Москве", [_span("Москве", "CITY", 8, 14)]),
        ],
    )
    assert load_done_ids(checkpoint) == {7}


def test_checkpoint_to_parquet_filters_dedupes_and_buckets(tmp_path):
    checkpoint = _write(
        tmp_path / "ck.jsonl",
        [
            _legacy_row(1, "stale row"),
            _current_row(2, "отели в Москве", [_span("Москве", "CITY", 8, 14)]),
            # Same request_id regenerated later in an append-only file: last wins.
            _current_row(2, "новые отели в Москве", [_span("Москве", "CITY", 14, 20)]),
            _current_row(
                3,
                "Москва, Россия",
                [_span("Москва", "CITY", 0, 6), _span("Россия", "COUNTRY", 8, 14)],
                valid=False,
            ),
            _current_row(
                4,
                "погода Ростов Ростовская область",
                [
                    _span("Ростов", "CITY", 7, 13),
                    _span("Ростовская область", "REGION", 14, 32),
                ],
                sample_source="city_admin1",
            ),
            _current_row(5, "old prompt", [_span("old", "CITY", 0, 3)], version=1),
        ],
    )
    out = tmp_path / "out.parquet"
    cfg = SimpleNamespace(checkpoint_path=checkpoint, output_path=str(out))

    assert checkpoint_to_parquet(cfg) == 2

    df = pl.read_parquet(out).sort("request_id")
    assert df["request_id"].to_list() == [2, 4]
    assert df["query"][0] == "новые отели в Москве"
    # `isolanguage` is renamed on the way out; `valid`/`invalid_reason` are internal.
    assert "language" in df.columns
    assert "valid" not in df.columns and "invalid_reason" not in df.columns
    # Gold buckets are typed per query type: one_city names no region, city_admin1 does.
    assert df["gold_city_entities"].to_list() == ["Москве", "Ростов"]
    assert df["gold_admin1_entities"].to_list() == ["", "Ростовская область"]
    assert df["gold_country_entities"].to_list() == ["", ""]
    # Offsets survive the round trip as a typed struct, not JSON text.
    assert df["entities"][1].to_list()[0] == {
        "text": "Ростов",
        "label": "CITY",
        "start": 7,
        "end": 13,
    }


def test_checkpoint_to_parquet_refuses_an_all_legacy_checkpoint(tmp_path):
    """Better a loud exit than a Parquet with a schema downstream no longer expects."""
    checkpoint = _write(tmp_path / "ck.jsonl", [_legacy_row(1, "stale")])
    cfg = SimpleNamespace(
        checkpoint_path=checkpoint, output_path=str(tmp_path / "out.parquet")
    )
    with pytest.raises(SystemExit, match="no prompt_version"):
        checkpoint_to_parquet(cfg)


def test_checkpoint_to_parquet_reads_past_a_long_legacy_prefix(tmp_path):
    """Current rows must survive more legacy rows than the schema-inference window.

    The checkpoint is append-only, so its *oldest* rows are the narrowest ones.
    Polars infers the schema from the first 100 rows by default, which silently
    drops `prompt_version` / `entities` / `confidence` from a file that begins
    with a long legacy prefix — and then the guard above fires on a file full of
    perfectly good rows.
    """
    legacy = [_legacy_row(i, f"stale {i}") for i in range(150)]
    current = [
        _current_row(1000 + i, "отели в Москве", [_span("Москве", "CITY", 8, 14)])
        for i in range(3)
    ]
    checkpoint = _write(tmp_path / "ck.jsonl", legacy + current)
    out = tmp_path / "out.parquet"
    cfg = SimpleNamespace(checkpoint_path=checkpoint, output_path=str(out))

    assert checkpoint_to_parquet(cfg) == 3
    assert pl.read_parquet(out)["confidence"].to_list() == [0.9] * 3


def test_checkpoint_row_persists_a_row_that_never_produced_a_body():
    """An unusable row is written as invalid, so it is paid for once, not per run.

    `checkpoint_to_parquet` drops `valid=False`, so nothing reaches the dataset —
    but the warm start now sees the row as done instead of retrying it forever.
    """
    from src.dataset.generate import Attempt

    row = _plan_fields(11, "one_city") | {"style": "search", "topic": "hotels"}
    out = _checkpoint_row(row, Attempt(None, "unparsable response: boom", 3))

    assert out["query"] == ""
    assert out["entities"] == []
    assert out["confidence"] == 0.0
    assert out["valid"] is False
    assert out["invalid_reason"] == "unparsable response: boom"
    assert out["pool"] == "all"


# ---------------------------------------------------------------------------
# The retry ladder
# ---------------------------------------------------------------------------


class _StubClient:
    """A client whose ``chat.completions.create`` replays canned responses."""

    def __init__(self, responses: list):
        self._responses = responses
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self.calls += 1
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def _response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=10,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )


def _generate(responses: list) -> tuple:
    client = _StubClient(responses)
    cfg = DatasetConfig()
    row = _plan_row("one_city") | {"request_id": 1}
    return client, generate_validated(client, cfg, row, UsageStats()), cfg


def test_an_exhausted_token_budget_is_not_retried():
    """The same prompt under the same budget fails identically — retrying pays twice."""
    client, result, _ = _generate([_response("", finish_reason="length")])

    assert client.calls == 1
    assert result.generation is None
    assert result.reason is not None and "budget" in result.reason


def test_a_malformed_body_is_retried_to_the_attempt_limit():
    client, result, cfg = _generate([_response("not json at all")])

    assert client.calls == cfg.max_generation_attempts
    assert result.generation is None
    assert result.reason is not None and "unparsable" in result.reason


def test_a_contract_failure_keeps_the_last_generation():
    """An invalid row is persisted with its reason rather than dropped."""
    body = json.dumps(
        {
            "query": "Ростов",
            "entities": [{"text": "Ростов", "label": "CITY"}],
            "confidence": 1.0,
        }
    )
    client, result, cfg = _generate([_response(body)])

    assert client.calls == cfg.max_generation_attempts
    assert result.generation is not None
    assert result.reason == "query is the bare name"


def test_a_retry_can_recover_the_row():
    good = json.dumps(
        {
            "query": "отели в Ростове",
            "entities": [{"text": "Ростове", "label": "CITY"}],
            "confidence": 0.9,
        }
    )
    client, result, _ = _generate([_response("nope"), _response(good)])

    assert client.calls == 2
    assert result.reason is None
    assert result.attempts == 2


def test_a_transport_error_propagates():
    """Transport failures belong to the run's retry pass, not to this loop."""
    with pytest.raises(ConnectionError):
        _generate([ConnectionError("connection reset")])
