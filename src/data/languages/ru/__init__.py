"""Russian language plugin.

Consolidates transliteration (to English), city description template,
and query templates with pymorphy3 inflection from the former
``src/data/russian/`` package.
"""
import functools
from .. import register_language


class RussianLanguage:
    """Russian language plugin implementing the LanguagePlugin protocol."""

    code: str = "ru"
    name: str = "Russian"

    # Lazy-initialised morphological analyser (expensive to create).
    @functools.cached_property
    def _morph(self):
        from pymorphy3 import MorphAnalyzer
        return MorphAnalyzer(lang="ru")

    # -- Morphological helpers ------------------------------------------------

    def _inflect_text(self, text: str, case: str) -> str:
        """Inflect every word in *text* to the given grammatical case."""
        words = text.split()
        is_upper = [w.istitle() for w in words]
        result: list[str] = []
        for up, word in zip(is_upper, words):
            parsed = self._morph.parse(word)[0]
            inflected = parsed.inflect({case})
            form = inflected.word if inflected else word
            if up:
                form = form.capitalize()
            result.append(form)
        return " ".join(result)

    # -- Query templates ------------------------------------------------------

    def templates_single_city(self, city: str) -> list[str]:
        """Return query variants for a bare city name (Russian)."""
        city_nomn = self._inflect_text(city, "nomn")
        city_gent = self._inflect_text(city, "gent")
        city_datv = self._inflect_text(city, "datv")
        city_accs = self._inflect_text(city, "accs")
        city_ablt = self._inflect_text(city, "ablt")
        city_loct = self._inflect_text(city, "loct")
        return [
            # -- short --
            city,
            f"Город {city_nomn}",
            f"погода {city_gent}",
            f"карта {city_gent}",
            f"население {city_gent}",
            f"история {city_gent}",
            f"фото {city_gent}",
            f"новости {city_gent}",
            f"афиша {city_gent}",
            f"работа в {city_loct}",
            # -- medium --
            f"какие парки и скверы есть в {city_loct}",
            f"музеи и галереи {city_gent} с бесплатным входом",
            f"театры и концертные залы в {city_loct}",
            f"стрит-арт и граффити в {city_loct}",
            f"блошиные рынки и антикварные лавки в {city_loct}",
            f"прогулка по {city_datv} — интересные маршруты",
            f"знакомство с {city_ablt} за один день",
            f"велопрокат и велодорожки в {city_loct}",
            f"бассейны и аквапарки в {city_loct}",
            f"катки и лыжные трассы рядом с {city_ablt}",
            f"зоопарк и ботанический сад в {city_loct}",
            f"квест-комнаты и развлечения в {city_loct}",
            f"детские площадки и парки аттракционов в {city_loct}",
            f"храмы и религиозные памятники в {city_loct}",
            f"архитектурные достопримечательности {city_gent}",
            f"лучшие точки для фотографий в {city_loct}",
            f"экология и качество воздуха в {city_loct}",
            f"где найти ветеринарную клинику в {city_loct}",
            f"круглосуточные аптеки и больницы в {city_loct}",
            f"автомойки и шиномонтаж в {city_loct}",
            f"химчистки и прачечные рядом с центром {city_gent}",
            f"салоны красоты и барбершопы в {city_loct}",
            f"путеводитель по {city_datv} для туристов",
            # -- long --
            f"{city_nomn} — хочу узнать, какой там климат и сколько примерно стоит снять жилье на неделю",
            f"{city_nomn} привлекает многих туристов, но я хотел бы узнать, как там живут местные жители"
            f" и какие районы считаются спальными",
            f"Подскажите, какие есть варианты общественного транспорта в {city_loct}"
            f" и как добраться из аэропорта до центра",
            f"Мне нужно найти стоматологическую клинику с хорошими отзывами в {city_loct},"
            f" желательно недалеко от центра",
            f"Хочу организовать день рождения в {city_loct} — подскажите необычные площадки"
            f" и заведения для праздников",
            f"Интересуюсь уличной едой и ночными рынками в {city_loct},"
            f" где можно недорого и вкусно перекусить поздно вечером",
            f"Планирую пробежку по набережной или парку, посоветуйте беговые маршруты в {city_loct}",
            f"Где в {city_loct} можно сдать вещи на переработку"
            f" и есть ли пункты приёма вторсырья рядом с центром",
            f"Собираюсь переехать и рассматриваю {city_accs} как один из вариантов,"
            f" расскажите про уровень жизни и инфраструктуру",
            f"Какие коворкинги и антикафе есть в {city_loct} с хорошим интернетом"
            f" и возможностью работать допоздна",
            f"Недавно услышал много хорошего про {city_accs} и хочу съездить туда"
            f" на пару дней — что стоит посмотреть в первую очередь",
            f"Можно ли в {city_loct} передвигаться без машины и насколько развит"
            f" общественный транспорт — метро, автобусы, трамваи",
            f"Где лучше остановиться в {city_loct} — в центре дороже, но удобнее,"
            f" или стоит выбрать спальный район подальше",
        ]

    def templates_single_city_country(
        self, city: str, country: str
    ) -> list[str]:
        """Return query variants for city + country (Russian)."""
        city_nomn = self._inflect_text(city, "nomn")
        city_gent = self._inflect_text(city, "gent")
        city_datv = self._inflect_text(city, "datv")
        city_accs = self._inflect_text(city, "accs")
        city_ablt = self._inflect_text(city, "ablt")
        city_loct = self._inflect_text(city, "loct")
        country_nomn = self._inflect_text(country, "nomn")
        country_gent = self._inflect_text(country, "gent")
        country_datv = self._inflect_text(country, "datv")
        country_accs = self._inflect_text(country, "accs")
        country_ablt = self._inflect_text(country, "ablt")
        country_loct = self._inflect_text(country, "loct")
        return [
            # -- short --
            f"{city_nomn}, {country_nomn}",
            f"Город {city_nomn}, {country_nomn}",
            f"виза в {city_accs}, {country_nomn}",
            f"валюта в {country_loct}, {city_nomn}",
            f"часовой пояс {city_gent}, {country_nomn}",
            f"язык общения в {city_loct}, {country_nomn}",
            f"праздники {country_gent}, {city_nomn}",
            f"посольство {country_gent} рядом с {city_ablt}",
            # -- medium --
            f"Где в {country_loct} находится {city_nomn}?",
            f"нужна ли виза для поездки в {city_accs}, {country_nomn}",
            f"какую валюту брать с собой в {city_accs}, {country_nomn}",
            f"какие прививки нужны перед поездкой в {city_accs}, {country_nomn}",
            f"культурные особенности и этикет в {city_loct}, {country_nomn}",
            f"размер чаевых и правила оплаты в {city_loct}, {country_nomn}",
            f"дресс-код и правила поведения в {city_loct}, {country_nomn}",
            f"можно ли пить воду из-под крана в {city_loct}, {country_nomn}",
            f"типы электрических розеток в {country_loct} и {city_loct}",
            f"номера экстренных служб в {city_loct}, {country_nomn}",
            f"SIM-карта и мобильный интернет в {city_loct}, {country_nomn}",
            f"стоимость роуминга в {country_loct} для приезжих в {city_accs}",
            f"законы о фото- и видеосъёмке в {country_loct}, {city_nomn}",
            f"правила провоза багажа по {country_datv} до {city_gent}",
            f"как работает такси и каршеринг в {city_loct}, {country_nomn}",
            f"международные школы и детские сады в {city_loct}, {country_nomn}",
            f"сообщества экспатов в {city_loct}, {country_nomn}",
            f"таможенные правила при въезде в {country_accs} через {city_accs}",
            f"возврат НДС для туристов в {country_loct}, покупки в {city_loct}",
            f"локальные соцсети и мессенджеры популярные в {city_loct}, {country_nomn}",
            f"медицинская страховка для поездки в {city_accs}, {country_nomn}",
            f"правила ввоза лекарств в {country_accs}, город {city_nomn}",
            # -- long --
            f"{city_nomn}, {country_nomn} — подскажите, какие документы нужны для въезда"
            f" и есть ли электронная виза",
            f"{city_nomn} в {country_loct} привлекает туристов со всего мира,"
            f" но мне хотелось бы узнать, как там обстоят дела с безопасностью",
            f"Планирую командировку в {country_accs} и мне нужно забронировать"
            f" гостиницу в {city_loct} на три ночи с трансфером из аэропорта",
            f"Какой часовой пояс действует в {country_loct} и во сколько открываются"
            f" магазины и банки в {city_loct}",
            f"Хочу оформить цифровую визу для удалённой работы из {city_gent},"
            f" {country_nomn}, — какие условия и сроки оформления",
            f"Подскажите, какие ограничения на ввоз алкоголя и табака действуют"
            f" в {country_loct} при прилёте в {city_accs}",
            f"Планирую длительное пребывание в {country_loct} и хочу открыть"
            f" банковский счёт в {city_loct} — какие банки работают с иностранцами",
            f"Какое водительское удостоверение нужно в {country_loct} и можно ли"
            f" арендовать машину в {city_loct} по российским правам",
            f"Слышал, что в {country_loct} строгие правила по использованию дронов —"
            f" можно ли запускать квадрокоптер в {city_loct}",
            f"Интересуюсь волонтёрскими программами в {country_loct},"
            f" есть ли что-то подходящее в {city_loct} на лето",
            f"путеводитель по {city_datv} — полезные советы перед поездкой"
            f" и знакомством с {country_ablt}",
        ]

    def templates_single_city_state(
        self, city: str, state: str
    ) -> list[str]:
        """Return query variants for city + state/region (Russian)."""
        city_nomn = self._inflect_text(city, "nomn")
        city_gent = self._inflect_text(city, "gent")
        city_datv = self._inflect_text(city, "datv")
        city_accs = self._inflect_text(city, "accs")
        city_ablt = self._inflect_text(city, "ablt")
        city_loct = self._inflect_text(city, "loct")
        state_nomn = self._inflect_text(state, "nomn")
        state_gent = self._inflect_text(state, "gent")
        state_datv = self._inflect_text(state, "datv")
        state_accs = self._inflect_text(state, "accs")
        state_loct = self._inflect_text(state, "loct")
        return [
            # -- short --
            f"{city_nomn}, {state_nomn}",
            f"Город {city_nomn}, {state_nomn}",
            f"погода {city_gent}, {state_nomn}",
            f"кухня {state_gent}, {city_nomn}",
            f"природа рядом с {city_ablt}, {state_nomn}",
            f"аэропорт {city_gent}, {state_nomn}",
            f"заповедники {state_gent} около {city_gent}",
            f"фестивали {state_gent} в {city_loct}",
            # -- medium --
            f"Как добраться до {city_gent} в {state_loct}",
            f"какие блюда региональной кухни попробовать в {city_loct}, {state_nomn}",
            f"пригородные электрички и автобусы из {city_gent}, {state_nomn}",
            f"соседние города рядом с {city_ablt} в {state_loct}",
            f"ярмарки и фермерские рынки в {city_loct}, {state_nomn}",
            f"агротуризм и фермы рядом с {city_ablt}, {state_nomn}",
            f"рыбалка и охота в окрестностях {city_gent}, {state_nomn}",
            f"кемпинги и места для палаток рядом с {city_ablt}, {state_nomn}",
            f"туристические тропы и пешие маршруты в {state_loct} около {city_gent}",
            f"термальные источники и санатории в {state_loct} недалеко от {city_gent}",
            f"краеведческий музей и исторические места {city_gent}, {state_nomn}",
            f"местные ремёсла и народные промыслы {state_gent}, {city_nomn}",
            f"конные прогулки и клубы верховой езды в {city_loct}, {state_nomn}",
            f"велосипедные маршруты по {state_datv} через {city_accs}",
            f"местные пивоварни и винодельни {state_gent} рядом с {city_ablt}",
            f"радиостанции и местные СМИ {city_gent}, {state_nomn}",
            f"спортивные команды и стадионы в {city_loct}, {state_nomn}",
            f"основные предприятия и заводы {city_gent} в {state_loct}",
            f"сезонная подработка в {city_loct}, {state_nomn}",
            f"волонтёрские организации {city_gent}, {state_nomn}",
            f"региональные традиции и обычаи {state_gent} в {city_loct}",
            f"автопрокат в {city_loct}, {state_nomn} — цены и условия",
            f"состояние дорог в {state_loct} на пути к {city_datv}",
            # -- long --
            f"{city_nomn} в {state_loct} — подскажите расписание автобусов"
            f" и как добраться из аэропорта до центра города",
            f"{city_nomn}, {state_nomn} — хочу узнать, какие промышленные предприятия"
            f" работают в этом районе и есть ли вакансии для инженеров",
            f"Хочу узнать среднюю стоимость аренды квартиры в {city_loct}, {state_nomn},"
            f" на месяц и какие есть спокойные районы для семьи",
            f"Планирую переезд в {state_accs} и рассматриваю несколько вариантов,"
            f" в том числе {city_accs} — расскажите про плюсы и минусы",
            f"Друзья рекомендовали съездить на выходные в {state_accs}, а именно"
            f" в {city_accs} — подскажите, что там стоит посмотреть",
            f"Какой ближайший аэропорт к {city_datv} в {state_loct} и есть ли"
            f" прямые рейсы из крупных городов",
            f"Ищу базу отдыха или загородный дом рядом с {city_ablt} в {state_loct}"
            f" для семейного отпуска на неделю",
            f"Насколько развита медицина и есть ли крупные больницы в {city_loct},"
            f" {state_nomn}, с современным оборудованием",
            f"В каком районе {state_gent} лучше всего жить — рассматриваю {city_accs}"
            f" и хочу узнать мнение местных жителей",
        ]

    def templates_single_city_state_country(
        self, city: str, state: str, country: str
    ) -> list[str]:
        """Return query variants for city + state + country (Russian)."""
        city_nomn = self._inflect_text(city, "nomn")
        city_gent = self._inflect_text(city, "gent")
        city_datv = self._inflect_text(city, "datv")
        city_accs = self._inflect_text(city, "accs")
        city_ablt = self._inflect_text(city, "ablt")
        city_loct = self._inflect_text(city, "loct")
        state_nomn = self._inflect_text(state, "nomn")
        state_gent = self._inflect_text(state, "gent")
        state_accs = self._inflect_text(state, "accs")
        state_loct = self._inflect_text(state, "loct")
        country_nomn = self._inflect_text(country, "nomn")
        country_gent = self._inflect_text(country, "gent")
        country_accs = self._inflect_text(country, "accs")
        country_loct = self._inflect_text(country, "loct")
        return [
            # -- short --
            f"{city_nomn}, {state_nomn}, {country_nomn}",
            f"Город {city_nomn}, {state_nomn}, {country_nomn}",
            f"индекс {city_gent}, {state_nomn}, {country_nomn}",
            f"климат {city_gent}, {state_nomn}, {country_nomn}",
            f"демография {city_gent}, {state_nomn}, {country_nomn}",
            f"коммуналка в {city_loct}, {state_nomn}, {country_nomn}",
            f"налоги в {city_loct}, {state_nomn}, {country_nomn}",
            f"полиция {city_gent}, {state_nomn}, {country_nomn}",
            # -- medium --
            f"почтовый индекс и формат адреса для {city_gent}, {state_nomn}, {country_nomn}",
            f"средняя температура по месяцам в {city_loct}, {state_nomn}, {country_nomn}",
            f"как зарегистрировать бизнес в {city_loct}, {state_nomn}, {country_nomn}",
            f"ставки подоходного налога в {country_loct}, {state_nomn}, для жителей {city_gent}",
            f"порядок записи ребёнка в школу в {city_loct}, {state_nomn}, {country_nomn}",
            f"тарифы на коммунальные услуги в {city_loct}, {state_nomn}, {country_nomn}",
            f"вывоз мусора и правила раздельного сбора в {city_loct}, {state_nomn}, {country_nomn}",
            f"как зарегистрировать автомобиль в {city_loct}, {state_nomn}, {country_nomn}",
            f"страхование квартиры в {city_loct}, {state_nomn}, {country_nomn}",
            f"депозит за аренду жилья в {city_loct}, {state_nomn}, {country_nomn}",
            f"мобильные операторы и тарифы в {city_loct}, {state_nomn}, {country_nomn}",
            f"интернет-провайдеры и скорость связи в {city_loct}, {state_nomn}, {country_nomn}",
            f"система отопления и расходы зимой в {city_loct}, {state_nomn}, {country_nomn}",
            f"уборка снега и коммунальные обязанности жителей {city_gent}, {state_nomn}, {country_nomn}",
            f"торговые центры и аутлеты рядом с {city_ablt}, {state_nomn}, {country_nomn}",
            f"благотворительные организации в {city_loct}, {state_nomn}, {country_nomn}",
            f"пожарная часть и спасательные службы {city_gent}, {state_nomn}, {country_nomn}",
            f"нотариальные услуги в {city_loct}, {state_nomn}, {country_nomn}",
            f"агентства недвижимости в {city_loct}, {state_nomn}, {country_nomn}",
            f"транспортные компании для переезда в {city_accs}, {state_nomn}, {country_nomn}",
            f"мебельные и строительные магазины в {city_loct}, {state_nomn}, {country_nomn}",
            f"автошколы и получение прав в {city_loct}, {state_nomn}, {country_nomn}",
            f"склады временного хранения вещей в {city_loct}, {state_nomn}, {country_nomn}",
            # -- long --
            f"{city_nomn}, {state_nomn}, {country_nomn} — подскажите, какой там климат"
            f" и средняя температура по сезонам",
            f"{city_nomn}, {state_nomn}, {country_nomn} — планирую переезд"
            f" и хочу заранее разобраться с пропиской и регистрацией по месту жительства",
            f"Нужно отправить посылку в {country_accs}, {state_accs}, в город {city_accs},"
            f" какой почтовый индекс и сроки доставки",
            f"Куда ещё можно съездить в {country_loct} после {city_gent} в {state_loct},"
            f" если осталось несколько свободных дней",
            f"Ищу контакты ближайшего отделения полиции и регистрационной палаты"
            f" в {city_loct}, {state_nomn}, {country_nomn}",
            f"Подскажите, какие документы нужны для оформления ребёнка"
            f" в детский сад или школу в {city_loct}, {state_nomn}, {country_nomn}",
            f"Собираюсь открыть небольшой бизнес в {country_loct} и рассматриваю"
            f" {city_accs} в {state_loct} как возможное место для офиса",
            f"Сколько в среднем стоят коммунальные услуги за двухкомнатную квартиру"
            f" зимой в {city_loct}, {state_nomn}, {country_nomn}",
            f"Какие риелторские агентства лучше всего помогут найти жильё"
            f" при переезде в {city_accs}, {state_nomn}, {country_nomn}",
            f"особенности региона {state_gent} в {country_gent} —"
            f" что нужно знать, переезжая в {city_datv}",
        ]

    # -- Two-city query templates ---------------------------------------------

    def templates_two_cities(self, city1: str, city2: str) -> list[str]:
        """Return query variants mentioning two cities (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1d = self._inflect_text(city1, "datv")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2d = self._inflect_text(city2, "datv")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        return [
            # -- short --
            f"{c1n} и {c2n}",
            f"{c1n} — {c2n}",
            f"{c1n} vs {c2n}",
            f"{c1n} или {c2n}",
            f"из {c1g} в {c2a}",
            f"между {c1i} и {c2i}",
            f"расстояние {c1n} {c2n}",
            f"маршрут {c1n} {c2n}",
            # -- medium --
            f"Как добраться из {c1g} в {c2a} на поезде или автобусе",
            f"Сколько лететь из {c1g} в {c2a} и какие есть рейсы",
            f"расстояние между {c1i} и {c2i} по трассе",
            f"время в пути из {c1g} в {c2a} на машине",
            f"автобусное расписание {c1n} — {c2n}",
            f"железнодорожные билеты из {c1g} в {c2a}",
            f"попутчики из {c1g} в {c2a}",
            f"где дешевле аренда — в {c1l} или в {c2l}",
            f"средняя зарплата в {c1l} и {c2l}",
            f"сравнение цен на продукты в {c1l} и {c2l}",
            f"экология и воздух в {c1l} по сравнению с {c2i}",
            f"уровень преступности в {c1l} и {c2l}",
            f"качество медицины в {c1l} или в {c2l}",
            f"где лучше школы — в {c1l} или в {c2l}",
            f"спортивная жизнь в {c1l} и {c2l}",
            f"ночная жизнь {c1g} или {c2g} — где интереснее",
            f"где чище и ухоженнее — в {c1l} или в {c2l}",
            f"парки и зелёные зоны в {c1l} и {c2l}",
            f"транспортная доступность {c1g} и {c2g}",
            f"музеи и театры в {c1l} и {c2l}",
            f"развлечения для детей в {c1l} и {c2l}",
            f"где вкуснее уличная еда — в {c1l} или в {c2l}",
            f"отзывы переехавших из {c1g} в {c2a}",
            # -- long --
            f"{c1n} или {c2n} — где лучше провести отпуск с семьёй и маленькими детьми",
            f"Сравните {c1a} и {c2a} по уровню жизни, инфраструктуре и транспорту",
            f"Планирую переезд и не могу решить между {c1i} и {c2i} —"
            f" подскажите, где комфортнее жить на зарплату среднего уровня",
            f"Хочу объехать оба города за одну поездку — как совместить {c1a} и {c2a}"
            f" и сколько дней заложить на каждый",
            f"Мне предложили работу одновременно в {c1l} и в {c2l} — подскажите,"
            f" где лучше условия для молодого специалиста без семьи",
            f"Какие плюсы и минусы жизни в {c1l} по сравнению с {c2i}:"
            f" климат, работа, досуг и стоимость жилья",
            f"Собираемся с друзьями на выходные и выбираем между {c1i} и {c2i} —"
            f" где больше развлечений и интересных мест",
            f"Интересно было бы узнать, как отличается стоимость коммунальных услуг"
            f" в {c1l} и {c2l} за одинаковую квартиру",
            f"Если ехать на машине из {c1g} в {c2a}, какие остановки по пути"
            f" стоит сделать и есть ли интересные места вдоль трассы",
            f"отправить посылку по {c1d} и {c2d} — какие службы доставки"
            f" работают между этими городами",
        ]

    def templates_two_cities_country(
        self, city1: str, country1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants for two cities, each with country (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        co1n = self._inflect_text(country1, "nomn")
        co1g = self._inflect_text(country1, "gent")
        co1a = self._inflect_text(country1, "accs")
        co1l = self._inflect_text(country1, "loct")
        co2n = self._inflect_text(country2, "nomn")
        co2g = self._inflect_text(country2, "gent")
        co2a = self._inflect_text(country2, "accs")
        co2i = self._inflect_text(country2, "ablt")
        co2l = self._inflect_text(country2, "loct")
        co1i = self._inflect_text(country1, "ablt")
        return [
            # -- short --
            f"{c1n} ({co1n}) и {c2n} ({co2n})",
            f"{c1n}, {co1n} — {c2n}, {co2n}",
            f"виза {c1n}, {co1n}, и {c2n}, {co2n}",
            f"валюта {co1g} и {co2g}, {c1n} и {c2n}",
            f"рейсы {c1n} {co1n} — {c2n} {co2n}",
            f"перелёт {c1n} — {c2n}, {co1n} — {co2n}",
            f"климат {c1g} ({co1n}) и {c2g} ({co2n})",
            f"цены {c1n} {co1n} vs {c2n} {co2n}",
            # -- medium --
            f"какие документы нужны для поездки из {c1g}, {co1n}, в {c2a}, {co2a}",
            f"разница в стоимости жизни между {c1i}, {co1n}, и {c2i}, {co2n}",
            f"визовые требования при перелёте из {c1g} ({co1n}) в {c2a} ({co2n})",
            f"разница в часовых поясах между {c1i}, {co1n}, и {c2i}, {co2n}",
            f"курс обмена валюты {co1g} и {co2g} для поездки {c1n} — {c2n}",
            f"авиакомпании с рейсами из {c1g}, {co1n}, в {c2a}, {co2n}",
            f"багажные нормы при перелёте {c1n} ({co1n}) — {c2n} ({co2n})",
            f"таможенные ограничения между {co1i} и {co2i}, {c1n} — {c2n}",
            f"прививки для поездки из {c1g} ({co1n}) в {c2a} ({co2n})",
            f"культурные различия между {c1i}, {co1n}, и {c2i}, {co2n}",
            f"телефонные коды и связь в {c1l}, {co1n}, и {c2l}, {co2n}",
            f"разница в налогах между {co1i} ({c1n}) и {co2i} ({c2n})",
            f"уровень безопасности в {c1l}, {co1n}, и в {c2l}, {co2n}",
            f"экспат-жизнь в {c1l} ({co1n}) по сравнению с {c2i} ({co2n})",
            f"стоимость медицинской страховки в {co1l} ({c1n}) и {co2l} ({c2n})",
            f"где лучше образование — в {c1l}, {co1n}, или {c2l}, {co2n}",
            f"сравнение пенсионных систем {co1g} и {co2g} для жителей {c1g} и {c2g}",
            f"какие продукты дешевле в {c1l}, {co1n}, а какие в {c2l}, {co2n}",
            f"международные перевозки между {c1i} ({co1n}) и {c2i} ({co2n})",
            f"разница в менталитете жителей {c1g}, {co1n}, и {c2g}, {co2n}",
            f"правила вождения в {co1l} ({c1n}) по сравнению с {co2i} ({c2n})",
            f"сравнение систем здравоохранения в {c1l}, {co1n}, и {c2l}, {co2n}",
            # -- long --
            f"Хочу посетить {c1a} в {co1l} и {c2a} в {co2l} за одну поездку,"
            f" как лучше спланировать маршрут и сколько денег взять",
            f"Сравните стоимость аренды квартиры и цены на продукты"
            f" в {c1l}, {co1n}, и в {c2l}, {co2n}",
            f"Планирую поездку из {c1g} ({co1n}) в {c2a} ({co2n}) и хочу узнать,"
            f" нужна ли транзитная виза и какие есть пересадки",
            f"Где зимой теплее и комфортнее — в {c1l}, {co1n}, или в {c2l}, {co2n},"
            f" учитывая влажность и ветер",
            f"Рассматриваю варианты для переезда и сравниваю {c1a} в {co1l} с {c2i}"
            f" в {co2l} по уровню жизни, работе и климату",
            f"Какой город лучше для фрилансера — {c1n} в {co1l} или {c2n} в {co2l}:"
            f" интернет, визы, стоимость жизни",
            f"Мне предложили вакансии и в {c1l} ({co1n}), и в {c2l} ({co2n}) —"
            f" подскажите, где выше покупательная способность зарплаты",
            f"Хочу сравнить систему общественного транспорта в {c1l}, {co1n},"
            f" и {c2l}, {co2n} — где удобнее жить без машины",
            f"Интересует качество воздуха и экологическая обстановка"
            f" в {c1l}, {co1n}, и {c2l}, {co2n}",
            f"Собираюсь на стажировку — выбираю между {c1i} в {co1l}"
            f" и {c2i} в {co2l}, где лучше условия для молодых специалистов",
            f"мечтаю объехать {co1a} и {co2a} — начну с {c1g} или {c2g}",
        ]

    def templates_two_cities_state(
        self, city1: str, state1: str, city2: str, state2: str
    ) -> list[str]:
        """Return query variants for two cities, each with state (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        s1n = self._inflect_text(state1, "nomn")
        s1g = self._inflect_text(state1, "gent")
        s1a = self._inflect_text(state1, "accs")
        s1l = self._inflect_text(state1, "loct")
        s2n = self._inflect_text(state2, "nomn")
        s2g = self._inflect_text(state2, "gent")
        s2a = self._inflect_text(state2, "accs")
        s2l = self._inflect_text(state2, "loct")
        return [
            # -- short --
            f"{c1n} ({s1n}) и {c2n} ({s2n})",
            f"{c1n}, {s1n} — {c2n}, {s2n}",
            f"школы {c1g} ({s1n}) и {c2g} ({s2n})",
            f"работа {c1n}, {s1n}, vs {c2n}, {s2n}",
            f"жильё {c1n} ({s1n}) и {c2n} ({s2n})",
            f"природа {s1g} ({c1n}) и {s2g} ({c2n})",
            f"промышленность {c1g}, {s1n}, и {c2g}, {s2n}",
            f"дороги {c1n} ({s1n}) — {c2n} ({s2n})",
            # -- medium --
            f"как доехать из {c1g}, {s1n}, в {c2a}, {s2a}, на общественном транспорте",
            f"сравнение аренды квартир в {c1l}, {s1n}, и в {c2l}, {s2n}",
            f"средние зарплаты в {c1l}, {s1n}, и {c2l}, {s2n}",
            f"уровень образования в {c1l} ({s1n}) по сравнению с {c2i} ({s2n})",
            f"экономика {s1g} ({c1n}) и {s2g} ({c2n}) — основные отрасли",
            f"сельское хозяйство {s1g} около {c1g} и {s2g} около {c2g}",
            f"медицинские учреждения {c1g}, {s1n}, и {c2g}, {s2n}",
            f"детские сады и ясли в {c1l}, {s1n}, и {c2l}, {s2n}",
            f"спортивные секции для детей в {c1l} ({s1n}) и {c2l} ({s2n})",
            f"культурные мероприятия в {c1l}, {s1n}, и {c2l}, {s2n}",
            f"торговые центры и магазины в {c1l} ({s1n}) и в {c2l} ({s2n})",
            f"общественный транспорт в {c1l}, {s1n}, по сравнению с {c2i}, {s2n}",
            f"экология и зелёные зоны в {c1l} ({s1n}) и {c2l} ({s2n})",
            f"криминогенная обстановка в {c1l}, {s1n}, и {c2l}, {s2n}",
            f"строительство нового жилья в {c1l} ({s1n}) и {c2l} ({s2n})",
            f"количество солнечных дней в {c1l}, {s1n}, по сравнению с {c2i}, {s2n}",
            f"фермерские рынки {c1g} ({s1n}) и {c2g} ({s2n})",
            f"туристическая привлекательность {c1g}, {s1n}, и {c2g}, {s2n}",
            f"ближайшие аэропорты к {c1i} ({s1n}) и к {c2i} ({s2n})",
            f"вузы и колледжи {c1g}, {s1n}, в сравнении с {c2i}, {s2n}",
            f"местные праздники и фестивали в {c1l} ({s1n}) и {c2l} ({s2n})",
            f"налоговая нагрузка для бизнеса в {c1l}, {s1n}, и {c2l}, {s2n}",
            f"стоимость бензина в {c1l} ({s1n}) по сравнению с {c2i} ({s2n})",
            # -- long --
            f"Выбираю для переезда между {c1i} в {s1l} и {c2i} в {s2l} —"
            f" подскажите, где лучше с работой и инфраструктурой",
            f"Сравните {c1a}, {s1n}, и {c2a}, {s2n}, по стоимости жилья,"
            f" качеству школ и доступности медицины",
            f"Мне предложили работу в {c1l} ({s1n}) и в {c2l} ({s2n}) —"
            f" в каком из городов выгоднее жить с семьёй",
            f"Хочу понять, какой из городов лучше подходит для пенсионеров —"
            f" {c1n} в {s1l} или {c2n} в {s2l}",
            f"Планирую купить дачу в {s1l} или в {s2l}, рассматриваю пригороды"
            f" {c1g} и {c2g} — где дешевле участки",
            f"Как различается стоимость коммунальных услуг зимой"
            f" в {c1l}, {s1n}, и в {c2l}, {s2n}",
            f"Дети заканчивают школу и выбираем вуз — какие перспективы"
            f" для студентов в {c1l} ({s1n}) и {c2l} ({s2n})",
            f"Рассматриваем с женой два варианта для жизни — {c1a} в {s1l}"
            f" или {c2a} в {s2l}, где спокойнее и чище",
            f"Интересует развитие IT-индустрии в {c1l}, {s1n},"
            f" по сравнению с {c2i}, {s2n} — где больше вакансий",
            f"путешествие через {s1a} из {c1g} в {c2a} — какие"
            f" достопримечательности посмотреть по дороге",
        ]

    def templates_two_cities_state_country(
        self, city1: str, state1: str, country1: str,
        city2: str, state2: str, country2: str,
    ) -> list[str]:
        """Return query variants for two cities, each with state + country (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        s1n = self._inflect_text(state1, "nomn")
        s1g = self._inflect_text(state1, "gent")
        s1l = self._inflect_text(state1, "loct")
        s2n = self._inflect_text(state2, "nomn")
        s2g = self._inflect_text(state2, "gent")
        s2l = self._inflect_text(state2, "loct")
        co1n = self._inflect_text(country1, "nomn")
        co1g = self._inflect_text(country1, "gent")
        co1l = self._inflect_text(country1, "loct")
        co2n = self._inflect_text(country2, "nomn")
        co2g = self._inflect_text(country2, "gent")
        co2i = self._inflect_text(country2, "ablt")
        co2l = self._inflect_text(country2, "loct")
        co1i = self._inflect_text(country1, "ablt")
        return [
            # -- short --
            f"{c1n}, {s1n}, {co1n} — {c2n}, {s2n}, {co2n}",
            f"рейс {c1n} ({s1n}, {co1n}) — {c2n} ({s2n}, {co2n})",
            f"климат {c1g}, {s1n}, {co1n}, и {c2g}, {s2n}, {co2n}",
            f"цены {c1n} ({co1n}) vs {c2n} ({co2n})",
            f"жизнь {c1n}, {co1n}, и {c2n}, {co2n}",
            f"зарплаты {c1g} ({co1n}) и {c2g} ({co2n})",
            f"аренда {c1n}, {s1n}, {co1n}, и {c2n}, {s2n}, {co2n}",
            # -- medium --
            f"как добраться из {c1g} ({s1n}, {co1n}) в {c2a} ({s2n}, {co2n})",
            f"авиабилеты из {c1g}, {s1n}, {co1n}, в {c2a}, {s2n}, {co2n}",
            f"сравнение климата {c1g} в {s1l}, {co1l}, и {c2g} в {s2l}, {co2l}",
            f"стоимость жизни в {c1l}, {s1n}, {co1n}, и в {c2l}, {s2n}, {co2n}",
            f"разница в уровне зарплат между {c1i} ({s1n}, {co1n}) и {c2i} ({s2n}, {co2n})",
            f"условия для удалённой работы в {c1l} ({co1n}) и {c2l} ({co2n})",
            f"система здравоохранения {c1g} ({s1n}, {co1n}) vs {c2g} ({s2n}, {co2n})",
            f"качество интернета в {c1l}, {s1n}, {co1n}, и {c2l}, {s2n}, {co2n}",
            f"безопасность в {c1l} ({s1n}, {co1n}) и в {c2l} ({s2n}, {co2n})",
            f"разница в налогах между {co1i} ({c1n}, {s1n}) и {co2i} ({c2n}, {s2n})",
            f"транспортная инфраструктура {c1g} ({s1n}, {co1n}) и {c2g} ({s2n}, {co2n})",
            f"возможности для бизнеса в {c1l}, {s1n}, {co1n}, и {c2l}, {s2n}, {co2n}",
            f"образовательные учреждения {c1g} ({co1n}) по сравнению с {c2i} ({co2n})",
            f"экологическая обстановка в {c1l}, {s1n}, {co1n}, и {c2l}, {s2n}, {co2n}",
            f"религиозное и культурное разнообразие {c1g} ({co1n}) и {c2g} ({co2n})",
            f"спортивная инфраструктура {c1g}, {s1n}, {co1n}, и {c2g}, {s2n}, {co2n}",
            f"парки и рекреационные зоны {c1g} ({s1n}, {co1n}) и {c2g} ({s2n}, {co2n})",
            f"стоимость медицинских услуг в {c1l} ({co1n}) и {c2l} ({co2n})",
            f"местная кухня {c1g} ({s1g}, {co1n}) и {c2g} ({s2g}, {co2n})",
            f"адаптация иммигрантов в {c1l} ({co1n}) по сравнению с {c2i} ({co2n})",
            f"правовая защита арендаторов в {co1l} ({c1n}) и {co2l} ({c2n})",
            # -- long --
            f"Сравните {c1a} ({s1n}, {co1n}) и {c2a} ({s2n}, {co2n}) по всем основным"
            f" параметрам — климат, работа, жильё, образование",
            f"Где лучше жить и работать удалённо — в {c1l} ({s1n}, {co1n})"
            f" или {c2l} ({s2n}, {co2n}), с учётом визовых программ",
            f"Планирую поездку и хочу посетить оба города — {c1a} в {s1l} ({co1n})"
            f" и {c2a} в {s2l} ({co2n}), как лучше построить маршрут",
            f"Ищу авиабилеты с пересадкой из {c1g}, {s1n}, {co1n},"
            f" в {c2a}, {s2n}, {co2n}, на ближайшие даты",
            f"Мне интересно, как отличается ритм жизни в {c1l} ({s1n}, {co1n})"
            f" от {c2g} ({s2n}, {co2n}) — темп, привычки, распорядок дня",
            f"Выбираю страну для жизни и сравниваю {c1a} в {s1l}, {co1n},"
            f" с {c2i} в {s2l}, {co2n} — где проще адаптироваться",
            f"Какой из городов подойдёт для семьи с двумя детьми —"
            f" {c1n}, {s1n}, {co1n}, или {c2n}, {s2n}, {co2n}",
            f"Хочу понять разницу в покупательной способности между {c1i}"
            f" ({s1n}, {co1n}) и {c2i} ({s2n}, {co2n}) при одинаковой зарплате",
            f"Рассматриваю {c1a} ({s1n}, {co1n}) и {c2a} ({s2n}, {co2n})"
            f" как варианты для открытия филиала компании",
            f"Собираю отзывы экспатов о жизни в {c1l}, {s1n}, {co1n},"
            f" и в {c2l}, {s2n}, {co2n} — делюсь опытом в блоге",
            f"Планирую маршрут — сначала {c1n} ({s1n}, {co1n}),"
            f" затем {c2n} ({s2n}, {co2n}), подскажите оптимальный транспорт",
            f"Сравните налоговую нагрузку для фрилансеров в {c1l} ({s1n}, {co1n})"
            f" и в {c2l} ({s2n}, {co2n}) — где выгоднее регистрировать ИП",
            f"виза {co1g} для жителей {co2g} — проще через {c1n} ({s1n})"
            f" или {c2n} ({s2n})",
        ]

    def templates_two_cities_one_country(
        self, city1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with country, city2 bare (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2d = self._inflect_text(city2, "datv")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        co1n = self._inflect_text(country1, "nomn")
        co1g = self._inflect_text(country1, "gent")
        co1a = self._inflect_text(country1, "accs")
        co1l = self._inflect_text(country1, "loct")
        return [
            # -- short --
            f"{c1n}, {co1n} и {c2n}",
            f"{c1n}, {co1n} — {c2n}",
            f"{c1n} ({co1n}) vs {c2n}",
            f"маршрут {c1n}, {co1n}, — {c2n}",
            f"билеты {c1n} ({co1n}) — {c2n}",
            f"расстояние {c1n}, {co1n}, до {c2g}",
            f"{c1n} ({co1n}) и {c2n} — карта",
            f"рейсы {c1n}, {co1n}, {c2n}",
            # -- medium --
            f"расстояние от {c1g} в {co1l} до {c2g} и как лучше добраться",
            f"ищу билеты из {c1g}, {co1n}, в {c2a} на ближайшие даты",
            f"автобусы и поезда из {c1g}, {co1n}, до {c2g}",
            f"попутчики из {c1g} ({co1n}) в {c2a}",
            f"время перелёта из {c1g}, {co1n}, до {c2g}",
            f"какой город крупнее — {c1n} в {co1l} или {c2n}",
            f"где теплее зимой — в {c1l} ({co1n}) или в {c2l}",
            f"стоимость жизни в {c1l}, {co1n}, и {c2l}",
            f"сравнение цен на жильё в {c1l} ({co1n}) и {c2l}",
            f"рынок труда {c1g}, {co1n}, по сравнению с {c2i}",
            f"достопримечательности {c1g} ({co1n}) и {c2g}",
            f"уличная еда в {c1l}, {co1n}, vs {c2l}",
            f"общественный транспорт {c1g}, {co1n}, и {c2g}",
            f"парки и набережные {c1g} ({co1n}) и {c2g}",
            f"ночные клубы и бары в {c1l}, {co1n}, и в {c2l}",
            f"музеи и выставки в {c1l} ({co1n}) и {c2l}",
            f"спортивные мероприятия в {c1l}, {co1n}, и {c2l}",
            f"климат {c1g} ({co1n}) и {c2g} по месяцам",
            f"куда лучше поехать с детьми — в {c1a} ({co1n}) или {c2a}",
            f"фестивали и мероприятия {c1g}, {co1n}, и {c2g}",
            f"кофейни и пекарни в {c1l} ({co1n}) и {c2l}",
            f"шопинг в {c1l}, {co1n}, по сравнению с {c2i}",
            f"велоинфраструктура {c1g} ({co1n}) и {c2g}",
            # -- long --
            f"Планирую поездку в {co1a} и хочу посетить {c1a} и {c2a} —"
            f" как лучше совместить два города в одном маршруте",
            f"Еду в {co1a} и выбираю, остановиться в {c1l} или в {c2l} —"
            f" где удобнее как база для экскурсий по окрестностям",
            f"Сравните {c1a} в {co1l} и {c2a} по количеству развлечений,"
            f" ресторанов и культурных мероприятий",
            f"Хочу провести неделю в {co1l} — три дня в {c1l} и четыре в {c2l},"
            f" подскажите, как распределить время",
            f"Мне предложили работу в {c1l} ({co1n}), но жена хочет жить в {c2l} —"
            f" можно ли ездить каждый день на работу между городами",
            f"Друзья рекомендовали {c1a} ({co1n}), а в интернете хвалят {c2a} —"
            f" где интереснее для первого визита",
            f"Перевожу бизнес из {c2g} в {c1a}, {co1n},"
            f" подскажите разницу в налогах и аренде офисов",
            f"Какой из городов лучше для студента — {c1n} в {co1l} или {c2n}:"
            f" стоимость общежития, подработка, развлечения",
            f"Ищу недвижимость и сравниваю рынок {c1g} в {co1l} с рынком {c2g} —"
            f" где выгоднее инвестировать в квартиру",
            f"туры из {co1g} к {c2d} — удобнее ехать через {c1i} или напрямую",
        ]

    def templates_two_cities_one_state(
        self, city1: str, state1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 bare (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        s1n = self._inflect_text(state1, "nomn")
        s1g = self._inflect_text(state1, "gent")
        s1a = self._inflect_text(state1, "accs")
        s1l = self._inflect_text(state1, "loct")
        return [
            # -- short --
            f"{c1n}, {s1n} и {c2n}",
            f"{c1n}, {s1n} — {c2n}",
            f"{c1n} ({s1n}) vs {c2n}",
            f"дорога {c1n} ({s1n}) — {c2n}",
            f"электричка {c1n}, {s1n}, — {c2n}",
            f"расстояние {c1n}, {s1n}, до {c2g}",
            f"погода {c1g} ({s1n}) и {c2g}",
            f"жильё {c1n}, {s1n}, и {c2n}",
            # -- medium --
            f"расстояние от {c1g} в {s1l} до {c2g} и сколько ехать на машине",
            f"есть ли автобусы из {c1g}, {s1n}, в {c2a}",
            f"электрички и поезда из {c1g} ({s1n}) в {c2a}",
            f"пробки на трассе между {c1i} ({s1n}) и {c2i}",
            f"попутчики из {c1g}, {s1n}, до {c2g}",
            f"бензин и заправки по дороге из {c1g} ({s1n}) в {c2a}",
            f"цены на недвижимость в {c1l}, {s1n}, по сравнению с {c2l}",
            f"стоимость аренды в {c1l} ({s1n}) и в {c2l}",
            f"средние зарплаты в {c1l}, {s1n}, и {c2l}",
            f"где лучше школы — в {c1l} ({s1n}) или в {c2l}",
            f"детские сады в {c1l}, {s1n}, и {c2l}",
            f"поликлиники и больницы {c1g} ({s1n}) и {c2g}",
            f"торговые центры и рынки в {c1l}, {s1n}, и {c2l}",
            f"спортзалы и фитнес-клубы в {c1l} ({s1n}) и {c2l}",
            f"кинотеатры и развлечения в {c1l}, {s1n}, и {c2l}",
            f"доставка еды в {c1l} ({s1n}) и {c2l}",
            f"такси из {c1g} ({s1n}) в {c2a} — стоимость и время",
            f"экология в {c1l}, {s1n}, по сравнению с {c2i}",
            f"где больше зелёных зон — в {c1l} ({s1n}) или {c2l}",
            f"водоснабжение и качество воды в {c1l}, {s1n}, и {c2l}",
            f"местные газеты и новостные порталы {c1g} ({s1n}) и {c2g}",
            f"культурная жизнь {c1g}, {s1n}, по сравнению с {c2i}",
            # -- long --
            f"Хочу посетить {c1a} в {s1l} и заодно заехать в {c2a} —"
            f" реально ли уложиться за один день на машине",
            f"Выбираю между {c1i} в {s1l} и {c2i} для покупки квартиры —"
            f" где перспективнее рынок недвижимости и лучше инфраструктура",
            f"Живу в {c1l} ({s1n}) и думаю перевезти семью в {c2a} —"
            f" подскажите, стоит ли переезд с точки зрения качества жизни",
            f"Работаю в {c1l}, {s1n}, а жена получила предложение в {c2l} —"
            f" можно ли ездить на работу между этими городами каждый день",
            f"Планирую автомобильную поездку из {c1g} ({s1n}) через {c2a}"
            f" и обратно — какие достопримечательности по пути",
            f"Мне нужно перевезти мебель из {c1g}, {s1n}, в {c2a} —"
            f" какие транспортные компании работают на этом направлении",
            f"Дети поступают в вуз — сравните возможности для студентов"
            f" в {c1l} ({s1n}) и {c2l}",
            f"Ищу дачу на полпути между {c1i} ({s1n}) и {c2i},"
            f" чтобы было удобно добираться из обоих городов",
            f"Какой из городов спокойнее для жизни на пенсии —"
            f" {c1n} в {s1l} или {c2n}, учитывая медицину и экологию",
            f"Сравните экологию и уровень шума в {c1l} ({s1n}) и {c2l}"
            f" — у меня маленький ребёнок и важна тихая обстановка",
            f"экскурсия по городам {s1g}: {c1n} и {c2n} — что успеть"
            f" за один выходной, объехав {s1a}",
        ]

    def templates_two_cities_one_state_country(
        self, city1: str, state1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state + country, city2 bare (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        s1n = self._inflect_text(state1, "nomn")
        s1g = self._inflect_text(state1, "gent")
        s1a = self._inflect_text(state1, "accs")
        co1n = self._inflect_text(country1, "nomn")
        co1g = self._inflect_text(country1, "gent")
        co1l = self._inflect_text(country1, "loct")
        return [
            # -- short --
            f"{c1n}, {s1n}, {co1n} и {c2n}",
            f"{c1n}, {s1n}, {co1n} — {c2n}",
            f"рейс {c1n} ({s1n}, {co1n}) — {c2n}",
            f"билеты {c1n}, {co1n}, — {c2n}",
            f"маршрут {c1n} ({co1n}) — {c2n}",
            f"километраж {c1n}, {s1n}, {co1n}, до {c2g}",
            f"климат {c1g} ({co1n}) и {c2g}",
            f"цены {c1n}, {co1n}, и {c2n}",
            # -- medium --
            f"расстояние от {c1g} ({s1n}, {co1n}) до {c2g} и как лучше добраться",
            f"есть ли прямые рейсы из {c1g}, {s1n}, {co1n}, в {c2a}",
            f"время перелёта из {c1g} ({s1n}, {co1n}) до {c2g}",
            f"автобусы или поезда из {c1g}, {s1n}, {co1n}, в {c2a}",
            f"сравнение климата {c1g} ({s1n}, {co1n}) и {c2g}",
            f"дешевле ли жить в {c1l}, {s1n}, {co1n}, чем в {c2l}",
            f"средние зарплаты в {c1l} ({s1n}, {co1n}) и {c2l}",
            f"стоимость аренды жилья в {c1l}, {s1n}, {co1n}, vs {c2l}",
            f"качество образования в {c1l} ({s1n}, {co1n}) и {c2l}",
            f"медицина и больницы {c1g} ({s1n}, {co1n}) vs {c2g}",
            f"общественный транспорт {c1g}, {s1n}, {co1n}, и {c2g}",
            f"безопасность в {c1l} ({s1n}, {co1n}) по сравнению с {c2i}",
            f"рестораны и кафе {c1g} ({s1n}, {co1n}) vs {c2g}",
            f"культурная жизнь в {c1l}, {s1n}, {co1n}, и {c2l}",
            f"природа и парки рядом с {c1i} ({s1n}, {co1n}) и {c2i}",
            f"шопинг в {c1l} ({s1n}, {co1n}) по сравнению с {c2i}",
            f"ночная жизнь {c1g}, {s1n}, {co1n}, и {c2g}",
            f"спортивная инфраструктура {c1g} ({co1n}) и {c2g}",
            f"коворкинги и удалённая работа в {c1l} ({co1n}) и {c2l}",
            f"экологическая обстановка {c1g} ({s1n}, {co1n}) и {c2g}",
            f"фестивали и массовые события {c1g}, {s1n}, {co1n}, и {c2g}",
            f"исторические достопримечательности {c1g} ({co1n}) vs {c2g}",
            f"возможности для фриланса в {c1l} ({s1n}, {co1n}) и {c2l}",
            # -- long --
            f"Планирую маршрут из {c1g} ({s1n}, {co1n}) до {c2g} на автомобиле —"
            f" подскажите оптимальную дорогу и интересные остановки",
            f"Рассматриваю переезд из {c1g}, {s1n}, {co1n}, в {c2a} —"
            f" стоит ли менять город с точки зрения карьеры и уровня жизни",
            f"Хочу отправить груз из {c1g} ({s1n}, {co1n}) в {c2a} —"
            f" какие транспортные компании предлагают лучшие тарифы",
            f"Сравните {c1a}, {s1n}, {co1n}, и {c2a} по возможностям"
            f" для молодой семьи с детьми — школы, медицина, досуг",
            f"Думаю открыть филиал компании и выбираю между {c1i} ({s1n}, {co1n})"
            f" и {c2i} — где выгоднее условия для бизнеса",
            f"Еду из {c1g}, {s1n}, {co1n}, и хочу по пути заехать"
            f" в {c2a} на пару дней — что стоит посмотреть",
            f"Как отличается ритм жизни в {c1l} ({s1n}, {co1n})"
            f" от ритма жизни в {c2l} — темп, привычки, менталитет",
            f"Ищу авиабилеты с багажом из {c1g}, {s1n}, {co1n}, в {c2a}"
            f" на ближайший месяц — какие авиакомпании летают",
            f"Насколько отличается стоимость продуктов в {c1l} ({s1n}, {co1n})"
            f" от цен в {c2l} — хочу спланировать бюджет на месяц",
            f"туры по городам {s1g}, {co1g} — {c1n} и {c2n} в одном маршруте",
            f"хочу объехать {s1a} в {co1l} — начну с {c1g}"
            f" и закончу путешествие в {c2l}",
        ]

    def templates_two_cities_state_and_country(
        self, city1: str, state1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 with country (Russian)."""
        c1n = self._inflect_text(city1, "nomn")
        c1g = self._inflect_text(city1, "gent")
        c1a = self._inflect_text(city1, "accs")
        c1i = self._inflect_text(city1, "ablt")
        c1l = self._inflect_text(city1, "loct")
        c2n = self._inflect_text(city2, "nomn")
        c2g = self._inflect_text(city2, "gent")
        c2a = self._inflect_text(city2, "accs")
        c2i = self._inflect_text(city2, "ablt")
        c2l = self._inflect_text(city2, "loct")
        s1n = self._inflect_text(state1, "nomn")
        s1g = self._inflect_text(state1, "gent")
        s1a = self._inflect_text(state1, "accs")
        s1l = self._inflect_text(state1, "loct")
        co2n = self._inflect_text(country2, "nomn")
        co2g = self._inflect_text(country2, "gent")
        co2a = self._inflect_text(country2, "accs")
        co2l = self._inflect_text(country2, "loct")
        return [
            # -- short --
            f"{c1n} ({s1n}) и {c2n} ({co2n})",
            f"{c1n}, {s1n} — {c2n}, {co2n}",
            f"{c1n} ({s1n}) vs {c2n} ({co2n})",
            f"рейсы {c1n}, {s1n} — {c2n}, {co2n}",
            f"билеты {c1n} ({s1n}) — {c2n} ({co2n})",
            f"климат {c1g} ({s1n}) и {c2g} ({co2n})",
            f"работа {c1n}, {s1n}, и {c2n}, {co2n}",
            f"аренда {c1n} ({s1n}) vs {c2n} ({co2n})",
            # -- medium --
            f"как добраться из {c1g} ({s1n}) до {c2g} в {co2l} и сколько это займёт",
            f"ищу билеты из {c1g} ({s1n}) в {c2a}, {co2a}, на следующую неделю",
            f"перелёт из {c1g}, {s1n}, в {c2a}, {co2n} — время и стоимость",
            f"визовые вопросы при поездке из {c1g} ({s1n}) в {c2a} ({co2n})",
            f"стоимость аренды жилья в {c1l}, {s1n}, и в {c2l}, {co2n}",
            f"где выше качество жизни — в {c1l} ({s1n}) или {c2l} ({co2n})",
            f"средние зарплаты в {c1l}, {s1n}, и {c2l}, {co2n}",
            f"вакансии в {c1l} ({s1n}) по сравнению с {c2l} ({co2n})",
            f"система образования {c1g}, {s1n}, и {c2g}, {co2n}",
            f"медицина и страховка в {c1l} ({s1n}) vs {c2l} ({co2n})",
            f"климат и погода {c1g}, {s1n}, по сравнению с {c2i}, {co2n}",
            f"транспортная система {c1g} ({s1n}) и {c2g} ({co2n})",
            f"безопасность в {c1l}, {s1n}, и в {c2l}, {co2n}",
            f"парки и зоны отдыха {c1g} ({s1n}) и {c2g} ({co2n})",
            f"культурная жизнь в {c1l}, {s1n}, по сравнению с {c2i}, {co2n}",
            f"местная кухня {c1g}, {s1n}, и {c2g}, {co2n}",
            f"шопинг и торговые центры в {c1l} ({s1n}) и {c2l} ({co2n})",
            f"интернет и мобильная связь {c1g}, {s1n}, vs {c2g}, {co2n}",
            f"ночная жизнь и развлечения {c1g} ({s1n}) и {c2g} ({co2n})",
            f"экология и чистота в {c1l}, {s1n}, и {c2l}, {co2n}",
            f"налоговая нагрузка в {c1l} ({s1n}) по сравнению с {c2l} ({co2n})",
            f"возможности для удалённой работы в {c1l}, {s1n}, и {c2l}, {co2n}",
            f"стоимость продуктовой корзины в {c1l} ({s1n}) и {c2l} ({co2n})",
            # -- long --
            f"Сравните {c1a} ({s1n}) и {c2a} ({co2n}) по уровню зарплат,"
            f" стоимости жилья и общей стоимости жизни",
            f"Рассматриваю переезд из {c1g}, {s1n}, в {c2a}, {co2n}, —"
            f" какие документы нужны и как проходит адаптация",
            f"Хочу понять, где лучше растить детей — в {c1l} ({s1n})"
            f" или {c2l} ({co2n}), учитывая школы, медицину и безопасность",
            f"Мне предложили контракт в {c2l}, {co2n}, а сейчас живу в {c1l}, {s1n} —"
            f" стоит ли переезжать ради карьерного роста",
            f"Планирую бизнес-поездку из {c1g} ({s1n}) в {c2a} ({co2n})"
            f" и хочу совместить с осмотром города — сколько дней заложить",
            f"Какой город комфортнее для пенсионеров — {c1n} в {s1l}"
            f" или {c2n} в {co2l}, с учётом медицины и стоимости жизни",
            f"Друзья зовут переехать в {c2a} ({co2n}), но я привык к {c1l}, {s1n} —"
            f" подскажите объективные плюсы и минусы обоих вариантов",
            f"Какие авиакомпании выполняют рейсы из {c1g}, {s1n},"
            f" в {c2a}, {co2n}, и как часто бывают распродажи билетов",
            f"Провожу анализ рынка труда для айтишников — сравниваю"
            f" вакансии в {c1l} ({s1n}) и в {c2l} ({co2n})",
            f"восхищаюсь {c1i} в {s1g} — но хочу сравнить с городами {co2g}",
            f"маршрут через {s1a} из {c1g} до {c2g}, {co2n} —"
            f" сколько займёт дорога на машине",
        ]

    def city_description(
        self,
        city: str,
        country: str,
        admin1_name: str,
        other_names: list[str],
    ) -> str:
        """Build a short Russian description of a city."""
        text = f"Город {city}, страна {country}, {admin1_name}"
        if other_names:
            text += f". Также называют: {', '.join(other_names)}"
        return text


# Auto-register on import
register_language(RussianLanguage())
