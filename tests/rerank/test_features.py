"""Unit tests for the reranker's span/document match features.

Pure functions over strings — no DB, no live API, no trained model.
"""

import pytest

from src.rerank.features import (
    FEATURE_COLUMNS,
    MATCH_FEATURES,
    NAME_SEPARATOR,
    build_row,
)


def _doc(names: list[str], country: str = "", admin1: str = "") -> str:
    """A ``build_descriptions``-shaped document.

    Mirrors the real builder, including the detail that places with no admin1
    keep only two lines (the join is stripped).
    """
    return "\n".join([NAME_SEPARATOR.join(names), country, admin1]).strip()


def _match(
    document: str, city: str = "", country: str = "", admin1: str = ""
) -> dict[str, float]:
    row = build_row(
        city_entities=city,
        country_entities=country,
        admin1_entities=admin1,
        document=document,
        population=1000,
        retriever_score=1.0,
        retriever_rank=0,
    )
    return {name: float(row[name]) for name in MATCH_FEATURES}


def test_build_row_emits_exactly_the_declared_columns():
    row = build_row("Москва", "Россия", "", _doc(["Москва"], "Russia"), 1, 1.0, 0)
    assert set(row) == set(FEATURE_COLUMNS)


class TestCityMatching:
    def test_declension_matches_fuzzily_but_not_exactly(self):
        """ "Литл-Роке" is the same place as "Литл-Рок" — a substring/n-gram hit,
        not an exact one. This is the case a word-level bag of words cannot see."""
        m = _match(_doc(["Little Rock", "Литл-Рок"], "United States"), city="Литл-Роке")
        assert m["city_ngram_containment"] == pytest.approx(1.0)
        assert m["city_substr_match"] == 1.0
        assert m["city_exact_match"] == 0.0

    def test_unrelated_candidate_scores_low(self):
        m = _match(_doc(["Rockville", "Роквилл"], "United States"), city="Литл-Роке")
        assert m["city_ngram_containment"] < 0.5
        assert m["city_substr_match"] == 0.0
        assert m["city_token_cover"] == 0.0

    def test_span_wider_than_the_place_it_names(self):
        """GLiNER tags "Van Gölü" (Lake Van) but the gold answer is the city Van,
        so the candidate name has to be matched *into* the span, not the reverse."""
        m = _match(_doc(["Van", "Ван", "凡城"], "Turkey", "Van"), city="Van Gölü")
        assert m["city_exact_match"] == 1.0
        assert m["city_token_cover"] == pytest.approx(0.5)

    def test_multi_city_query_partially_covered_by_one_candidate(self):
        """A two-city query still fully matches each city, but each candidate
        accounts for only half of the spans."""
        m = _match(
            _doc(["Kazan", "Казань"], "Russia", "Tatarstan"), city="Москва Казань"
        )
        assert m["city_exact_match"] == 1.0
        assert m["city_token_cover"] == pytest.approx(0.5)

    def test_cjk_names_match_whole(self):
        """CJK names are shorter than the n-gram size, so they are compared whole."""
        m = _match(_doc(["Shanghai", "上海市"], "China", "Shanghai"), city="上海市")
        assert m["city_exact_match"] == 1.0
        assert m["city_ngram_containment"] == pytest.approx(1.0)

    def test_extra_spellings_do_not_dilute_the_match(self):
        """The regression the ``NAME_SEPARATOR`` change fixes.

        Space-joining the spellings collapsed them into one pseudo-name, so only a
        1/n fraction of its n-grams could ever sit inside a span naming the place
        once — a place scored *lower* the better documented it was. Matched per
        name, the extra spellings cost nothing.
        """
        lean = _match(_doc(["Литл-Рок"]), city="Литл-Роке")
        rich = _match(
            _doc(["Литл-Рок", "Little Rock", "小岩城", "Litl-Rok"]), city="Литл-Роке"
        )
        assert rich["city_ngram_containment"] == pytest.approx(
            lean["city_ngram_containment"]
        )
        assert rich["city_ngram_containment"] == pytest.approx(1.0)


class TestCountryMatching:
    @pytest.mark.parametrize("span", ["Россия", "Rusya", "俄罗斯", "Russia"])
    def test_country_resolves_across_languages(self, span):
        """The document's country line is always English; the span is in the
        query's language, so both go through an ISO code before comparison."""
        m = _match(_doc(["Kazan"], "Russia", "Tatarstan"), city="Kazan", country=span)
        assert m["country_match"] == 1.0
        assert m["has_country_span"] == 1.0

    def test_homonym_in_the_wrong_country_is_marked(self):
        """ "上海市 美国" asks for the US Shanghai — the Chinese one must not match."""
        m = _match(_doc(["Shanghai", "上海市"], "China"), city="上海市", country="美国")
        assert m["city_exact_match"] == 1.0
        assert m["country_match"] == 0.0

    def test_homonym_in_the_named_country_matches(self):
        m = _match(
            _doc(["Shanghai", "上海市"], "United States", "West Virginia"),
            city="上海市",
            country="美国",
        )
        assert m["country_match"] == 1.0

    def test_absent_span_is_distinct_from_a_mismatch(self):
        """-1.0 means "no comparison possible", which is not the same state as
        "the countries disagree" (0.0) and must land in a different leaf."""
        m = _match(_doc(["Kazan"], "Russia"), city="Kazan")
        assert m["country_match"] == -1.0
        assert m["has_country_span"] == 0.0

    def test_country_name_must_match_whole_tokens(self):
        """ "China" inside "Chinatown" is not a country mention."""
        m = _match(_doc(["Chinatown"], "United States"), city="Chinatown")
        assert m["country_match"] == -1.0


class TestAdmin1Matching:
    def test_english_region_matches(self):
        m = _match(
            _doc(["Tucson", "Тусон"], "United States", "Arizona"),
            city="Tucson",
            admin1="Arizona",
        )
        assert m["admin1_substr_match"] == 1.0
        assert m["admin1_ngram_containment"] == pytest.approx(1.0)
        assert m["has_admin1_span"] == 1.0

    def test_non_latin_region_is_a_known_miss(self):
        """admin1 names exist in English only (``admin1CodesASCII.txt``), so a
        Russian span cannot match. ``has_admin1_span`` keeps this distinguishable
        from "no region was named" — see the note in :func:`build_row`."""
        m = _match(
            _doc(["Shenzhen", "Шэньчжэнь"], "China", "Guangdong"),
            city="Шэньчжэнь",
            admin1="Гуандун",
        )
        assert m["admin1_ngram_containment"] == 0.0
        assert m["has_admin1_span"] == 1.0


class TestDocumentShapes:
    def test_two_line_document_has_no_admin1(self):
        document = _doc(["Van", "Ван", "凡城"], "Turkey")
        assert document.count("\n") == 1
        m = _match(document, city="Van", admin1="Van")
        assert m["city_exact_match"] == 1.0
        assert m["admin1_substr_match"] == 0.0
        assert m["admin1_ngram_containment"] == 0.0

    def test_empty_document_scores_zero_without_raising(self):
        """The online reranker scores candidates with no known description
        against an empty document rather than dropping them."""
        m = _match("", city="Москва", country="Россия", admin1="Москва")
        assert m["city_ngram_containment"] == 0.0
        assert m["country_match"] == -1.0
        assert m["has_country_span"] == 1.0

    def test_turkish_dotless_i_folds_onto_ascii(self):
        """ "Iğdır" and the ASCII spelling GeoNames stores must normalise alike."""
        m = _match(_doc(["Igdir", "Iğdır"], "Turkey"), city="Iğdır")
        assert m["city_exact_match"] == 1.0
