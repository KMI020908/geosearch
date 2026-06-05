"""Abstract interface (Protocol) that every language plugin must implement."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguagePlugin(Protocol):
    """Structural interface for a language module."""

    code: str
    name: str

    # -- Query templates ------------------------------------------------------

    def templates_single_city(self, city: str) -> list[str]: ...

    def templates_single_city_country(
        self, city: str, country: str
    ) -> list[str]: ...

    def templates_single_city_state(
        self, city: str, state: str
    ) -> list[str]: ...

    def templates_single_city_state_country(
        self, city: str, state: str, country: str
    ) -> list[str]: ...

    # -- Two-city query templates ---------------------------------------------

    def templates_two_cities(
        self, city1: str, city2: str
    ) -> list[str]: ...

    def templates_two_cities_country(
        self, city1: str, country1: str, city2: str, country2: str
    ) -> list[str]: ...

    def templates_two_cities_state(
        self, city1: str, state1: str, city2: str, state2: str
    ) -> list[str]: ...

    def templates_two_cities_state_country(
        self, city1: str, state1: str, country1: str,
        city2: str, state2: str, country2: str,
    ) -> list[str]: ...

    def templates_two_cities_one_country(
        self, city1: str, country1: str, city2: str
    ) -> list[str]: ...

    def templates_two_cities_one_state(
        self, city1: str, state1: str, city2: str
    ) -> list[str]: ...

    def templates_two_cities_one_state_country(
        self, city1: str, state1: str, country1: str, city2: str
    ) -> list[str]: ...

    def templates_two_cities_state_and_country(
        self, city1: str, state1: str, city2: str, country2: str
    ) -> list[str]: ...

    # -- Transliteration ------------------------------------------------------

    def transliterate_to(self, text: str, target_lang: str) -> str | None: ...
