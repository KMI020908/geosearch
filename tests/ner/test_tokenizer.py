"""Unit tests for the CJK-aware word splitter and span alignment.

Pure functions over strings — no model, no GPU, no data files.

These cover the assumption the whole NER pipeline rests on: GLiNER classifies
*token* spans, so a gold span that does not land on token boundaries is one the
model can neither be trained on nor predict.
"""

from src.ner.tokenizer import CjkAwareSplitter, align_span, tokenize


def _tokens(text: str) -> list[str]:
    return [token for token, _, _ in tokenize(text)]


class TestSplitting:
    def test_han_runs_split_per_ideograph(self):
        """The reason this splitter exists: 莫斯科 inside 莫斯科新闻 must be reachable.

        Under GLiNER's own rule `\\w+` matches Han, so the whole run is one token
        and the city span is structurally unpredictable.
        """
        assert _tokens("莫斯科新闻") == ["莫", "斯", "科", "新", "闻"]

    def test_latin_and_cyrillic_words_stay_whole(self):
        assert _tokens("рейсы Moscow") == ["рейсы", "Moscow"]

    def test_turkish_diacritics_do_not_split_a_word(self):
        assert _tokens("Iğdır hava") == ["Iğdır", "hava"]

    def test_hyphenated_words_stay_one_token(self):
        """GLiNER's own rule keeps `-`/`_` joins together; we inherit it."""
        assert _tokens("Ростов-на-Дону") == ["Ростов-на-Дону"]

    def test_punctuation_is_its_own_token(self):
        assert _tokens("Москва, Казань") == ["Москва", ",", "Казань"]

    def test_supplementary_plane_ideographs_split_too(self):
        """𠯫 (U+20BEB) is outside the BMP and appears in this dataset's names."""
        assert _tokens("𠯫村") == ["𠯫", "村"]

    def test_mixed_script_is_segmented_on_both_rules(self):
        assert _tokens("上海 news") == ["上", "海", "news"]

    def test_offsets_index_back_into_the_source_text(self):
        text = "рейсы 上海"
        for token, start, end in tokenize(text):
            assert text[start:end] == token

    def test_splitter_is_callable_as_a_generator(self):
        """GLiNER assigns this straight onto `data_processor.words_splitter`."""
        assert list(CjkAwareSplitter()("ab")) == [("ab", 0, 2)]


class TestAlignSpan:
    def test_whole_token_span_maps_to_inclusive_indices(self):
        text = "рейсы в Москву"
        start = text.index("Москву")
        assert align_span(tokenize(text), start, start + len("Москву")) == (2, 2)

    def test_multi_token_span_covers_the_whole_range(self):
        text = "上海天气"
        assert align_span(tokenize(text), 0, 2) == (0, 1)

    def test_span_starting_inside_a_token_is_rejected(self):
        """Not snapped outwards: a widened span teaches a labelling error.

        Tagging 莫斯科新闻 ("Moscow news") as a city is wrong, not approximately
        right, so the row is dropped instead.
        """
        text = "Москвы"
        assert align_span(tokenize(text), 0, 5) is None

    def test_span_ending_inside_a_token_is_rejected(self):
        text = "рейсы Moscow"
        assert align_span(tokenize(text), 6, 9) is None

    def test_reversed_span_is_rejected(self):
        text = "a b"
        assert align_span(tokenize(text), 2, 1) is None

    def test_han_spans_align_under_this_splitter(self):
        """The measured failure the splitter fixed: 56/56 zh spans, now 0."""
        text = "莫斯科新闻"
        assert align_span(tokenize(text), 0, 3) == (0, 2)
