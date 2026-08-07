"""Unit tests for the GLiNER training-data export.

In-memory polars frames only — no Parquet, no DB, no model.

The two behaviours worth pinning: a row whose spans cannot be represented as
token indices is dropped whole rather than partially labelled, and the train/val
split never puts one place name on both sides.
"""

import polars as pl

from src.ner.dataset import build_example, build_examples, split_by_name


def _entity(text: str, label: str, start: int, end: int) -> dict:
    return {"text": text, "label": label, "start": start, "end": end}


def _row(
    query: str = "рейсы в Москву",
    entities: list[dict] | None = None,
    *,
    language: str = "ru",
    name: str = "Москва",
    request_id: int = 1,
) -> dict:
    if entities is None:
        entities = [_entity("Москву", "CITY", 8, 14)]
    return {
        "query": query,
        "entities": entities,
        "language": language,
        "sample_source": "one_city",
        "name": name,
        "request_id": request_id,
    }


class TestBuildExample:
    def test_aligned_span_becomes_inclusive_token_indices(self):
        example = build_example(_row())
        assert example is not None
        assert example["tokenized_text"] == ["рейсы", "в", "Москву"]
        assert example["ner"] == [[2, 2, "CITY"]]

    def test_raw_text_and_offsets_are_kept_for_evaluation(self):
        """`src.ner.evaluate` scores on raw text, so the export must carry it."""
        example = build_example(_row())
        assert example is not None
        assert example["text"] == "рейсы в Москву"
        assert example["entities"] == [_entity("Москву", "CITY", 8, 14)]
        assert example["language"] == "ru"

    def test_row_without_entities_is_dropped(self):
        assert build_example(_row(entities=[])) is None

    def test_row_whose_offsets_do_not_match_the_text_is_dropped(self):
        """A corrupt row — the offsets name a different substring than the span."""
        bad = _row(entities=[_entity("Москву", "CITY", 0, 6)])
        assert build_example(bad) is None

    def test_row_with_a_span_off_token_boundaries_is_dropped_whole(self):
        """Not partially labelled: an unlabelled city teaches the opposite lesson."""
        row = _row(
            query="Москвы новости",
            entities=[
                _entity("Москв", "CITY", 0, 5),
                _entity("новости", "CITY", 7, 14),
            ],
        )
        assert build_example(row) is None

    def test_chinese_spans_align(self):
        """The CJK splitter is what makes this expressible at all."""
        example = build_example(
            _row(
                query="莫斯科新闻",
                entities=[_entity("莫斯科", "CITY", 0, 3)],
                language="zh",
                name="莫斯科",
            )
        )
        assert example is not None
        assert example["ner"] == [[0, 2, "CITY"]]

    def test_build_examples_skips_the_unusable_rows(self):
        df = pl.DataFrame([_row(request_id=1), _row(entities=[], request_id=2)])
        assert len(build_examples(df)) == 1


class TestSplitByName:
    def _examples(self, names: list[str], language: str = "ru") -> list[dict]:
        return [
            {"name": name, "language": language, "ner": [], "text": name}
            for name in names
        ]

    def test_no_name_appears_on_both_sides(self):
        """The point of splitting by name: a val query must not be a paraphrase
        of a train query about the same place."""
        examples = self._examples([f"city{i}" for i in range(10)] * 3)
        train, val = split_by_name(examples, val_size=0.3, seed=42)
        assert {ex["name"] for ex in train} & {ex["name"] for ex in val} == set()

    def test_every_row_lands_on_exactly_one_side(self):
        examples = self._examples([f"city{i}" for i in range(10)])
        train, val = split_by_name(examples, val_size=0.3, seed=42)
        assert len(train) + len(val) == len(examples)

    def test_split_is_deterministic_for_a_seed(self):
        examples = self._examples([f"city{i}" for i in range(20)])
        first = split_by_name(examples, val_size=0.2, seed=7)
        second = split_by_name(examples, val_size=0.2, seed=7)
        assert [ex["name"] for ex in first[1]] == [ex["name"] for ex in second[1]]

    def test_each_language_is_held_out_separately(self):
        """Drawing val globally lets one language dominate the holdout by chance."""
        examples = self._examples([f"ru{i}" for i in range(10)], "ru")
        examples += self._examples([f"zh{i}" for i in range(10)], "zh")
        _, val = split_by_name(examples, val_size=0.3, seed=42)
        assert {ex["language"] for ex in val} == {"ru", "zh"}
