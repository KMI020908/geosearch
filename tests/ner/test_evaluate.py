"""Unit tests for the span-level NER metrics.

Scoring only — the model is never loaded. `score_predictions` takes the predicted
span sets directly, so the arithmetic can be pinned without a GPU or a checkpoint.
"""

import pytest

from src.ner.evaluate import Score, gold_spans, score_predictions

LABELS = ["CITY", "COUNTRY"]


def _example(text: str, entities: list[tuple[int, int, str]], language: str) -> dict:
    return {
        "text": text,
        "language": language,
        "entities": [
            {"text": text[s:e], "label": label, "start": s, "end": e}
            for s, e, label in entities
        ],
    }


class TestScore:
    def test_perfect_prediction(self):
        score = Score(tp=4, fp=0, fn=0)
        assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)

    def test_no_predictions_gives_zero_precision_not_a_crash(self):
        score = Score(tp=0, fp=0, fn=3)
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0

    def test_empty_bucket_is_all_zero(self):
        """A label absent from the split must not divide by zero."""
        score = Score(tp=0, fp=0, fn=0)
        assert score.f1 == 0.0
        assert score.gold == score.predicted == 0

    def test_f1_is_the_harmonic_mean(self):
        score = Score(tp=1, fp=1, fn=3)
        assert score.precision == pytest.approx(0.5)
        assert score.recall == pytest.approx(0.25)
        assert score.f1 == pytest.approx(1 / 3)

    def test_support_is_carried_alongside_the_rates(self):
        """A rate without its support reads noise as signal on ~47 val queries."""
        score = Score(tp=2, fp=1, fn=3)
        assert score.gold == 5
        assert score.predicted == 3

    def test_as_dict_is_json_serialisable(self):
        assert Score(1, 1, 1).as_dict() == {
            "gold": 2,
            "predicted": 2,
            "tp": 1,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }


def test_gold_spans_reads_character_offsets():
    example = _example("рейсы в Москву", [(8, 14, "CITY")], "ru")
    assert gold_spans(example) == {(8, 14, "CITY")}


class TestScorePredictions:
    def test_exact_match_counts_as_a_hit(self):
        data = [_example("рейсы в Москву", [(8, 14, "CITY")], "ru")]
        report = score_predictions(data, [{(8, 14, "CITY")}], LABELS)
        assert report["overall"].f1 == 1.0

    def test_a_span_off_by_a_word_is_not_partial_credit(self):
        """Downstream the span text is matched against candidate names, so a
        near-miss buys nothing there either."""
        data = [_example("рейсы в Москву", [(8, 14, "CITY")], "ru")]
        report = score_predictions(data, [{(0, 14, "CITY")}], LABELS)
        assert report["overall"].tp == 0
        assert report["overall"].fp == 1
        assert report["overall"].fn == 1

    def test_right_span_wrong_label_is_both_a_miss_and_a_false_positive(self):
        data = [_example("Россия", [(0, 6, "COUNTRY")], "ru")]
        report = score_predictions(data, [{(0, 6, "CITY")}], LABELS)
        assert report["label:COUNTRY"].fn == 1
        assert report["label:CITY"].fp == 1

    def test_averaging_is_micro_over_spans_not_macro_over_queries(self):
        """A query naming three cities feeds three names into retrieval, so it
        must weigh three times as much as one naming a single city."""
        data = [
            _example("Москва Казань Пермь", [(0, 6, "CITY"), (7, 13, "CITY")], "ru"),
            _example("Пермь", [(0, 5, "CITY")], "ru"),
        ]
        # Both spans right in the first query, the single span wrong in the second.
        report = score_predictions(
            data, [{(0, 6, "CITY"), (7, 13, "CITY")}, set()], LABELS
        )
        assert report["overall"].recall == pytest.approx(2 / 3)

    def test_language_buckets_split_by_the_row_language(self):
        data = [
            _example("Москва", [(0, 6, "CITY")], "ru"),
            _example("上海", [(0, 2, "CITY")], "zh"),
        ]
        report = score_predictions(data, [{(0, 6, "CITY")}, set()], LABELS)
        assert report["lang:ru"].f1 == 1.0
        assert report["lang:zh"].f1 == 0.0

    def test_every_declared_label_gets_a_bucket_even_when_absent(self):
        """Otherwise a report silently changes shape when a rare label misses
        the holdout, and two runs stop being comparable."""
        data = [_example("Москва", [(0, 6, "CITY")], "ru")]
        report = score_predictions(data, [{(0, 6, "CITY")}], LABELS)
        assert "label:COUNTRY" in report
        assert report["label:COUNTRY"].gold == 0

    def test_a_length_mismatch_is_an_error_not_a_silent_truncation(self):
        data = [_example("Москва", [(0, 6, "CITY")], "ru")]
        with pytest.raises(ValueError):
            score_predictions(data, [], LABELS)
