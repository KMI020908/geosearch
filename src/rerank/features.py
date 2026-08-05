"""Shared reranker feature builder — the single source of truth for both the
offline training set (:mod:`src.rerank.dataset`) and online scoring
(:mod:`src.rerank.model`).

Keeping one builder guarantees the model sees *identical* features at train and
serve time. This matters concretely here: the engine feeds the reranker the NER
spans, not the raw query — and it feeds them **split by entity type**, so the
training set must featurise the same typed buckets, not the full query text.

**Why the match features exist.** CatBoost ``text_features`` builds a bag of
words per column *independently*, so it cannot compute the intersection of
``city_entities`` and ``document``: learning that would take a pair of splits
("token X in the span AND token X in the document") for every token, from ~130
training groups. Measured on the mined data, the model gave all three entity
buckets 1.5% of its importance combined and ended up a noisy copy of
``retriever_rank`` — worse than the retriever's own order. Yet the signal is
large and trivially computable: a city token occurs in the document for 84.7% of
positives against 15.9% of negatives. So the overlap is computed here, in the
numeric features below, instead of being left for the trees to rediscover.

To change the feature set, edit the lists below and :func:`build_row`. To change
how GLiNER labels map onto the entity buckets, edit :data:`LABEL_BUCKETS`.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from country_list import countries_for_language

# The reranker's text features are the NER spans split by type plus the candidate
# ``document``. The three entity buckets are keyed exactly by their column names,
# so :data:`LABEL_BUCKETS`, :func:`bucket_entities`, and :func:`build_row` all
# stay in lock-step with this list.
TEXT_FEATURES = ["city_entities", "country_entities", "admin1_entities", "document"]

# Explicit query-span/candidate overlap, computed by :func:`build_row` (see the
# module docstring for why the trees cannot derive these from the text features).
MATCH_FEATURES = [
    "city_exact_match",
    "city_substr_match",
    "city_ngram_containment",
    "city_token_cover",
    "country_match",
    "has_country_span",
    "admin1_substr_match",
    "admin1_ngram_containment",
    "has_admin1_span",
]

NUMERIC_FEATURES = [
    "log_population",
    "retriever_score",
    "retriever_rank",
    *MATCH_FEATURES,
]
FEATURE_COLUMNS = TEXT_FEATURES + NUMERIC_FEATURES

# Map GLiNER labels (``settings.ner_labels``) onto the entity feature columns.
# GeoNames admin1 is the state/province/region level, so REGION and STATE both
# feed the single ``admin1_entities`` bucket — matching the candidate document,
# which also carries one admin1 line.
LABEL_BUCKETS: dict[str, set[str]] = {
    "city_entities": {"CITY"},
    "country_entities": {"COUNTRY"},
    "admin1_entities": {"REGION", "STATE"},
}

# The query dataset stores the *reference* spans in the same buckets under a
# ``gold_`` prefix (:mod:`src.dataset.generate`), so they are byte-comparable to
# the live ``entity_buckets``. Derived from :data:`LABEL_BUCKETS` rather than
# spelled out, so adding a bucket does not need a second edit here.
GOLD_BUCKET_COLS: dict[str, str] = {name: f"gold_{name}" for name in LABEL_BUCKETS}

# Separator between a candidate's name spellings on the first line of its
# ``document`` (:func:`src.rerank.dataset.build_descriptions`). A plain space
# loses the boundary between names — "Little Rock Литл-Рок" is indistinguishable
# from one four-token name — which rules out exact matching and, worse, turns the
# n-gram features into an inverse-popularity signal: one pseudo-name built from
# every spelling only ever has a 1/n fraction of its n-grams inside a span that
# names the place once, so a well-documented city scores *below* a village with a
# single name. Separated, each spelling is matched on its own and the best wins.
NAME_SEPARATOR = " | "

# Character n-gram size for the fuzzy containment features. Three is the usual
# choice for latin/cyrillic and is what tolerates declension ("литл роке" still
# covers every trigram of "литл рок"). Strings shorter than this are compared
# whole, which is what keeps CJK working: names like "上海市" are 2-3 characters.
_NGRAM_SIZE = 3

# Turkish dotless i does not decompose under NFKD and does not casefold to "i",
# so it would never match the ASCII spelling GeoNames stores. Every other
# in-scope diacritic (ğ ş ç ö ü, and Cyrillic й ё) is handled by NFKD.
_TRANSLATE = str.maketrans({"ı": "i", "İ": "i"})


def bucket_entities(spans: list[tuple[str, str]]) -> dict[str, str]:
    """Group ``(text, label)`` NER spans into the three entity feature strings.

    Spans sharing a bucket are space-joined in occurrence order. Every bucket key
    is always present (empty string when no span of that type was found), so the
    returned dict is safe to splat straight into :func:`build_row`. Labels not in
    :data:`LABEL_BUCKETS` are ignored.
    """
    buckets: dict[str, list[str]] = {name: [] for name in LABEL_BUCKETS}
    for text, label in spans:
        for name, labels in LABEL_BUCKETS.items():
            if label in labels:
                buckets[name].append(text)
                break
    return {name: " ".join(texts) for name, texts in buckets.items()}


@lru_cache(maxsize=1 << 16)
def _norm(text: str) -> str:
    """Casefold, strip diacritics and punctuation, collapse whitespace.

    Both sides of every comparison go through this, so the transliteration is
    only required to be *consistent*, not faithful: folding "ё"/"й" onto "е"/"и"
    loses nothing and buys tolerance for spelling variation.
    """
    folded = unicodedata.normalize("NFKD", text.casefold().translate(_TRANSLATE))
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in stripped).split())


def _ngrams(text: str) -> frozenset[str]:
    """Character n-grams of a normalised string; short strings are kept whole."""
    if not text:
        return frozenset()
    if len(text) <= _NGRAM_SIZE:
        return frozenset({text})
    return frozenset(
        text[i : i + _NGRAM_SIZE] for i in range(len(text) - _NGRAM_SIZE + 1)
    )


def _containment(needle: frozenset[str], haystack: frozenset[str]) -> float:
    """Fraction of ``needle``'s n-grams present in ``haystack``.

    Containment rather than Jaccard: the two sides differ in length by design (a
    candidate name against the query's whole span string), and Jaccard would
    penalise that difference hard enough to bury the signal.
    """
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


@lru_cache(maxsize=1)
def _country_codes_by_name() -> dict[str, str]:
    """Normalised country name (in any in-scope language) to ISO 3166-1 alpha-2.

    The candidate's document carries the *English* country name while the NER
    span is in the query's language, so "Россия" / "Rusya" / "俄罗斯" have to be
    resolved through a shared key before they can be compared to "Russia".
    English is inserted first and :meth:`dict.setdefault` keeps the first
    spelling, so an ambiguous name resolves the way the document spells it.
    """
    from src.config import settings

    mapping: dict[str, str] = {}
    for language in ("en", *settings.languages):
        for code, name in countries_for_language(language):
            mapping.setdefault(_norm(name), code)
    # ``build_descriptions`` falls back to the bare country code when a country
    # has no localised name, so the code itself has to resolve too.
    for code in set(mapping.values()):
        mapping.setdefault(_norm(code), code)
    return mapping


@lru_cache(maxsize=1 << 12)
def _resolve_countries(normalised: str) -> frozenset[str]:
    """Every country code named anywhere inside a normalised span string.

    Matching is on whole tokens so that "china" in "chinatown" does not count,
    and multi-token country names ("united states") are tried as a whole.
    """
    if not normalised:
        return frozenset()
    padded = f" {normalised} "
    return frozenset(
        code for name, code in _country_codes_by_name().items() if f" {name} " in padded
    )


@dataclass(frozen=True)
class _Candidate:
    """The parts of a candidate's ``document`` the match features compare against."""

    names: tuple[str, ...]
    name_ngrams: tuple[frozenset[str], ...]
    country_codes: frozenset[str]
    admin1: str
    admin1_ngrams: frozenset[str]


@lru_cache(maxsize=1 << 16)
def _parse_document(document: str) -> _Candidate:
    """Split a ``build_descriptions`` document into normalised comparable parts.

    The document is ``names / country / admin1``, one per line — but places with
    no admin1 have the trailing blank line stripped off, so anything from one to
    three lines has to parse.
    """
    lines = document.split("\n")
    names = tuple(
        normalised
        for raw in lines[0].split(NAME_SEPARATOR)
        if (normalised := _norm(raw))
    )
    admin1 = _norm(lines[2]) if len(lines) > 2 else ""
    return _Candidate(
        names=names,
        name_ngrams=tuple(_ngrams(name) for name in names),
        country_codes=_resolve_countries(_norm(lines[1]) if len(lines) > 1 else ""),
        admin1=admin1,
        admin1_ngrams=_ngrams(admin1),
    )


@dataclass(frozen=True)
class _Query:
    """The normalised query spans; constant across one query's candidates."""

    city: str
    city_tokens: tuple[str, ...]
    city_ngrams: frozenset[str]
    country_codes: frozenset[str]
    has_country: bool
    admin1: str
    admin1_ngrams: frozenset[str]
    has_admin1: bool


@lru_cache(maxsize=1 << 12)
def _parse_spans(city: str, country: str, admin1: str) -> _Query:
    """Normalise the three entity buckets once per query."""
    city_norm = _norm(city)
    admin1_norm = _norm(admin1)
    return _Query(
        city=city_norm,
        city_tokens=tuple(city_norm.split()),
        city_ngrams=_ngrams(city_norm),
        country_codes=_resolve_countries(_norm(country)),
        has_country=bool(country.strip()),
        admin1=admin1_norm,
        admin1_ngrams=_ngrams(admin1_norm),
        has_admin1=bool(admin1.strip()),
    )


def _match_features(query: _Query, candidate: _Candidate) -> dict[str, float]:
    """Overlap between the query's spans and one candidate's document.

    Every city measure runs candidate-name-against-span-string, not the reverse.
    That direction is what survives the two shapes the bucket strings actually
    take: several spans joined together ("Москва Казань" for a multi-city query,
    where only one of them is this candidate) and one span wider than the place
    it names ("Van Gölü", whose gold answer is the city ``Van``). Containment the
    other way round would be diluted in the first case and capped in the second.
    """
    padded_city = f" {query.city} "
    exact = substr = 0.0
    containment = 0.0
    for name, ngrams in zip(candidate.names, candidate.name_ngrams, strict=True):
        if f" {name} " in padded_city:
            exact = substr = 1.0
        elif name and name in query.city:
            substr = 1.0
        containment = max(containment, _containment(ngrams, query.city_ngrams))

    # Fraction of the spans that this candidate accounts for: 1.0 when the query
    # named exactly this place, 1/n when it named n cities and this is one.
    covered = sum(
        any(token in name for name in candidate.names) for token in query.city_tokens
    )
    token_cover = covered / len(query.city_tokens) if query.city_tokens else 0.0

    # -1.0 marks "no comparison was possible" — no span, or neither side
    # resolved — which is a different state from "the countries disagree" and
    # gets its own leaf. ``has_country_span`` disambiguates the two.
    if query.country_codes and candidate.country_codes:
        country_match = float(bool(query.country_codes & candidate.country_codes))
    else:
        country_match = -1.0

    return {
        "city_exact_match": exact,
        "city_substr_match": substr,
        "city_ngram_containment": containment,
        "city_token_cover": token_cover,
        "country_match": country_match,
        "has_country_span": float(query.has_country),
        "admin1_substr_match": float(
            bool(candidate.admin1) and candidate.admin1 in query.admin1
        ),
        "admin1_ngram_containment": _containment(
            candidate.admin1_ngrams, query.admin1_ngrams
        ),
        "has_admin1_span": float(query.has_admin1),
    }


def build_row(
    city_entities: str,
    country_entities: str,
    admin1_entities: str,
    document: str,
    population: int,
    retriever_score: float,
    retriever_rank: int,
) -> dict[str, object]:
    """Build the feature dict for one (query, candidate) pair.

    * ``city_entities`` / ``country_entities`` / ``admin1_entities`` — the NER
      spans of each type, space-joined (see :func:`bucket_entities`). Constant
      across a query's candidates; the ``document`` is what varies within a group.
    * ``document`` — the candidate's ``build_descriptions`` text.
    * ``log_population`` — ``log1p(population)``; population is the dominant
      tiebreak signal, log-scaled so trees split it cleanly.
    * ``retriever_score`` / ``retriever_rank`` — the BM25 score and 0-based
      position in the retrieved list.
    * :data:`MATCH_FEATURES` — explicit span/document overlap, city and country
      and admin1 separately (:func:`_match_features`).

    Known limitation: ``admin1_name`` comes from ``admin1CodesASCII.txt`` and
    exists **in English only**, while the span is in the query's language. The
    admin1 features therefore work for en/tr and read 0 for ru/zh ("Гуандун"
    against ``Guangdong``). Localised region names are not available — regions
    are ``feature_class='A'`` and the ETL loads only ``'P'``, so their
    ``alternate_name`` rows were never ingested. ``has_admin1_span`` at least
    lets the model tell a real 0 from a missing comparison.
    """
    query = _parse_spans(city_entities, country_entities, admin1_entities)
    candidate = _parse_document(document)
    return {
        "city_entities": city_entities,
        "country_entities": country_entities,
        "admin1_entities": admin1_entities,
        "document": document,
        "log_population": math.log1p(max(population, 0)),
        "retriever_score": float(retriever_score),
        "retriever_rank": float(retriever_rank),
        **_match_features(query, candidate),
    }
