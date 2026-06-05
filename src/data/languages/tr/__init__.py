"""Turkish language plugin.

Consolidates vowel harmony constants, transliteration (to English/ASCII),
city description template, and query templates with case suffixes from the
former ``src/data/turkish/`` package.
"""
from .. import register_language

# -- Turkish morphology constants ---------------------------------------------

BACK_VOWELS: str = "aıou"
FRONT_VOWELS: str = "eiöü"
VOICELESS: str = "çfhkpsşt"


class TurkishLanguage:
    """Turkish language plugin implementing the LanguagePlugin protocol."""

    code: str = "tr"
    name: str = "Turkish"

    # -- Morphological helpers ------------------------------------------------

    @staticmethod
    def _last_vowel_is_front(word: str) -> bool:
        """Return True if the last vowel in *word* is a front vowel."""
        for c in reversed(word):
            if c.lower() in FRONT_VOWELS:
                return True
            if c.lower() in BACK_VOWELS:
                return False
        return False

    @staticmethod
    def _inflect_text(text: str, case: str) -> str:
        """Append a Turkish case suffix to the last word of *text*.

        Applies e-type vowel harmony, voiceless consonant rule (suffix
        ``-t-`` after ``çfhkpsşt``), and apostrophe before the suffix
        for proper nouns.

        *case* must be one of ``"loc"`` (locative ``-da/-de``),
        ``"dat"`` (dative ``-a/-e``), ``"abl"`` (ablative ``-dan/-den``).
        """
        words = text.split()
        if not words:
            return text
        last = words[-1]
        if not last:
            return text
        front = TurkishLanguage._last_vowel_is_front(last)
        last_char = last[-1].lower()
        voiceless = last_char in VOICELESS

        if case == "loc":
            suffix = ("te" if front else "ta") if voiceless else ("de" if front else "da")
        elif case == "dat":
            if last_char in BACK_VOWELS + FRONT_VOWELS or last_char in "aeiou":
                suffix = "ye" if front else "ya"
            else:
                suffix = "e" if front else "a"
        elif case == "abl":
            suffix = ("ten" if front else "tan") if voiceless else ("den" if front else "dan")
        else:
            return text

        inflected_last = last + "'" + suffix
        if len(words) == 1:
            return inflected_last
        return " ".join(words[:-1] + [inflected_last])

    # -- Query templates ------------------------------------------------------

    def templates_single_city(self, city: str) -> list[str]:
        """Return query variants for a bare city name (Turkish)."""
        city_loc = self._inflect_text(city, "loc")
        city_dat = self._inflect_text(city, "dat")
        city_abl = self._inflect_text(city, "abl")
        return [
            # -- short --
            city,
            f"{city} şehri",
            f"{city_loc} hava durumu",
            f"{city} haritası",
            f"{city} nüfusu",
            f"{city} tarihi",
            f"{city} fotoğrafları",
            f"{city} haberleri",
            f"{city_loc} etkinlikler",
            f"{city_loc} iş ilanları",
            # -- medium --
            f"{city_loc} parklar ve yeşil alanlar",
            f"{city_loc} ücretsiz müzeler ve galeriler",
            f"{city_loc} tiyatrolar ve konser salonları",
            f"{city_loc} sokak sanatı ve grafiti noktaları",
            f"{city_loc} bit pazarları ve antikacılar",
            f"{city_loc} yürüyüş turları ve ilginç rotalar",
            f"{city} şehrini bir günde keşfet",
            f"{city_loc} bisiklet kiralama ve bisiklet yolları",
            f"{city_loc} yüzme havuzları ve su parkları",
            f"{city} yakınlarında buz pateni pistleri ve kayak merkezleri",
            f"{city_loc} hayvanat bahçesi ve botanik bahçesi",
            f"{city_loc} kaçış odaları ve eğlence mekanları",
            f"{city_loc} çocuk oyun alanları ve lunaparklar",
            f"{city_loc} camiler ve dini yapılar",
            f"{city} mimari öne çıkan yapılar",
            f"{city_loc} en iyi fotoğraf noktaları",
            f"{city_loc} hava kalitesi ve çevre verileri",
            f"{city_loc} veteriner klinikleri",
            f"{city_loc} 24 saat açık eczaneler ve hastaneler",
            f"{city_loc} oto yıkama ve lastikçiler",
            f"{city} şehir merkezinde kuru temizlemeciler ve çamaşırhaneler",
            f"{city_loc} kuaförler ve berberler",
            f"{city_dat} gelen turistler için rehber",
            # -- long --
            f"{city} — iklimi nasıl ve haftalık konaklama"
            f" ücretleri ne kadar öğrenmek istiyorum",
            f"{city} birçok turisti çekiyor ancak yerel halkın nasıl yaşadığını"
            f" ve hangi semtlerin konut bölgesi olduğunu bilmek istiyorum",
            f"{city_loc} toplu taşıma seçenekleri nelerdir"
            f" ve havalimanından şehir merkezine nasıl gidilir",
            f"{city_loc} iyi yorumları olan bir diş kliniği"
            f" bulmam gerekiyor, tercihen şehir merkezine yakın",
            f"{city_loc} doğum günü partisi düzenlemek istiyorum — sıra dışı"
            f" mekanlar ve kutlama yerleri önerir misiniz",
            f"{city_loc} sokak yemekleri ve gece pazarları ilgimi çekiyor,"
            f" akşam geç saatlerde nerede ucuz ve lezzetli yemek bulunur",
            f"{city_loc} sahil kenarında veya parkta koşu yapmayı planlıyorum,"
            f" koşu rotası önerir misiniz",
            f"{city_loc} geri dönüşüm için eşyaları nereye bırakabilirim"
            f" ve merkezde toplama noktası var mı",
            f"Taşınmayı düşünüyorum ve {city} şehrini seçeneklerden biri olarak değerlendiriyorum,"
            f" yaşam standardı ve altyapısı hakkında bilgi verin",
            f"{city_loc} iyi internetli ve geç saatlere kadar çalışılabilecek"
            f" ortak çalışma alanları ve anti-kafeler hangileri",
            f"{city} hakkında çok iyi şeyler duydum ve birkaç günlüğüne"
            f" gitmek istiyorum — önce ne görmeliyim",
            f"{city_loc} arabası olmadan dolaşılabilir mi ve toplu taşıma"
            f" ne kadar gelişmiş — metro, otobüs, tramvay",
            f"{city_loc} nerede kalmak daha iyi — merkez daha pahalı ama uygun,"
            f" yoksa daha uzak sakin bir semt mi tercih etmeliyim",
            f"{city_abl} çevredeki şehirlere günübirlik gezi rotaları"
            f" ve ulaşım seçenekleri neler",
        ]

    def templates_single_city_country(
        self, city: str, country: str
    ) -> list[str]:
        """Return query variants for city + country (Turkish)."""
        city_loc = self._inflect_text(city, "loc")
        city_dat = self._inflect_text(city, "dat")
        city_abl = self._inflect_text(city, "abl")
        country_loc = self._inflect_text(country, "loc")
        country_dat = self._inflect_text(country, "dat")
        country_abl = self._inflect_text(country, "abl")
        return [
            # -- short --
            f"{city}, {country}",
            f"{city} şehri, {country}",
            f"{city_dat} vize, {country}",
            f"{country_loc} para birimi, {city}",
            f"{city} saat dilimi, {country}",
            f"{city_loc} konuşulan dil, {country}",
            f"{country} tatil günleri, {city}",
            f"{city} yakınlarında {country} büyükelçiliği",
            # -- medium --
            f"{country_loc} {city} nerede bulunur?",
            f"{city_dat} seyahat için vize gerekiyor mu, {country}",
            f"{city_dat} hangi para birimini götürmeliyim, {country}",
            f"{city_dat} seyahat öncesi hangi aşılar gerekli, {country}",
            f"{city_loc} kültürel normlar ve görgü kuralları, {country}",
            f"{city_loc} bahşiş ve ödeme kuralları, {country}",
            f"{city_loc} kıyafet kuralları ve davranış kuralları, {country}",
            f"{city_loc} musluk suyu içilebilir mi, {country}",
            f"{country_loc} ve {city_loc} elektrik prizi tipleri",
            f"{city_loc} acil durum telefon numaraları, {country}",
            f"{city_loc} SIM kart ve mobil internet, {country}",
            f"{country_loc} ziyaretçiler için dolaşım maliyetleri, {city}",
            f"{country_loc} fotoğraf ve video çekim yasaları, {city}",
            f"{country_dat} {city_dat} bagaj kuralları",
            f"{city_loc} taksi ve araç paylaşımı nasıl çalışır, {country}",
            f"{city_loc} uluslararası okullar ve anaokulları, {country}",
            f"{city_loc} gurbetçi toplulukları, {country}",
            f"{country_dat} giriş sırasında gümrük kuralları, {city} üzerinden",
            f"{country_loc} turistler için KDV iadesi, {city_loc} alışveriş",
            f"{city_loc} popüler yerel sosyal ağlar ve mesajlaşma uygulamaları, {country}",
            f"{city_dat} seyahat sağlık sigortası, {country}",
            f"{country_dat} ilaç ithalatı kuralları, {city} şehri",
            # -- long --
            f"{city}, {country} — giriş için hangi belgeler gerekiyor"
            f" ve e-vize seçeneği var mı",
            f"{country_loc} {city} dünyanın dört bir yanından turistleri çekiyor,"
            f" ancak yalnız seyahat edenler için ne kadar güvenli bilmek istiyorum",
            f"{country_dat} iş seyahati planlıyorum ve {city_loc}"
            f" üç gecelik havalimanı transferli otel ayırtmam gerekiyor",
            f"{country_loc} hangi saat dilimi geçerli ve {city_loc}"
            f" mağazalar ve bankalar saat kaçta açılıyor",
            f"{city_abl}, {country}, uzaktan çalışmak için dijital göçebe vizesi"
            f" başvurusu yapmak istiyorum — koşullar ve süre nedir",
            f"{country_dat} varışta alkol ve tütün getirme"
            f" kısıtlamaları nelerdir, {city_dat} iniş yapıldığında",
            f"{country_loc} uzun süreli kalış planlıyorum ve {city_loc}"
            f" banka hesabı açmak istiyorum — hangi bankalar yabancılara hizmet veriyor",
            f"{country_loc} hangi tür ehliyet gerekli ve {city_loc}"
            f" mevcut ehliyetimle araba kiralayabilir miyim",
            f"{country_loc} katı drone düzenlemeleri olduğunu duydum —"
            f" {city_loc} quadcopter uçurulabilir mi",
            f"{country_loc} gönüllü programlarla ilgileniyorum,"
            f" yaz için {city_loc} uygun bir şey var mı",
            f"{country_abl} {city_dat} seyahat rehberi — faydalı"
            f" ipuçları ve tanışma önerileri",
        ]

    def templates_single_city_state(
        self, city: str, state: str
    ) -> list[str]:
        """Return query variants for city + state/region (Turkish)."""
        city_loc = self._inflect_text(city, "loc")
        city_dat = self._inflect_text(city, "dat")
        city_abl = self._inflect_text(city, "abl")
        state_loc = self._inflect_text(state, "loc")
        state_dat = self._inflect_text(state, "dat")
        state_abl = self._inflect_text(state, "abl")
        return [
            # -- short --
            f"{city}, {state}",
            f"{city} şehri, {state}",
            f"{city_loc} oteller, {state}",
            f"{city_loc} hava durumu, {state}",
            f"{city} posta kodu, {state}",
            f"{city} demografisi, {state}",
            f"{city_loc} suç oranı, {state}",
            f"{city_loc} okullar, {state}",
            # -- medium --
            f"{state_loc} {city_dat} nasıl gidilir",
            f"{city_loc} otobüs seferleri ve havalimanı transferi, {state}",
            f"{city_loc} bir yatak odalı daire ortalama kirası, {state}",
            f"{state_dat} taşınmayı düşünüyorum — {city} seçeneklerden biri",
            f"{city} şehir merkezinde restoranlar ve kafeler, {state}",
            f"{city}, {state}, aileler için güvenli mi",
            f"{city_loc} üniversiteler ve kolejler, {state}",
            f"hafta sonu {city_loc} yapılacak şeyler, {state}",
            f"{city_loc} yaşam hakkında bilgi verin, {state}",
            f"{city_loc} iş piyasası ve işe alım trendleri, {state}",
            f"{city_loc} spor salonları ve spor tesisleri, {state}",
            f"{city_loc} en iyi pastaneler ve kahveciler, {state}",
            f"{city_loc} kütüphaneler ve toplum merkezleri, {state}",
            f"{city_loc} çiftçi pazarları ve yerel ürünler, {state}",
            f"{city} yakınlarında yürüyüş parkurları ve doğa parkları, {state}",
            f"{city_loc} evcil hayvan dostu parklar ve veteriner klinikleri, {state}",
            f"{city_loc} toplu taşıma kapsamı, {state}",
            f"{city_loc} emlak vergileri ve ev sahibi maliyetleri, {state}",
            f"{city_loc} hızlı internetli ortak çalışma alanları, {state}",
            f"{city_loc} kreş ve anaokulu seçenekleri, {state}",
            f"{city_loc} hastaneler ve acil servisler, {state}",
            f"{city_loc} oto galeri ve tamirciler, {state}",
            f"{city_loc} yıllık festivaller ve toplum etkinlikleri, {state}",
            f"{city_loc} geri dönüşüm ve çöp toplama kuralları, {state}",
            # -- long --
            f"{city_loc} aylık kiralık daire fiyatlarını ve en iyi semtleri"
            f" öğrenmek istiyorum, {state}",
            f"{state_dat} taşınmayı planlıyorum ve {city} dahil birkaç şehri"
            f" değerlendiriyorum — artıları ve eksileri neler",
            f"{city_loc} küçük çocuklu aileler için en iyi okul bölgeleri"
            f" hangileri, {state}",
            f"{city_loc} bahçeli ve toplu taşımaya erişimi iyi bir ev arıyorum,"
            f" {state} — önerileriniz var mı",
            f"{city_loc} girişimcilik ortamı ve teknoloji sektörü nasıl,"
            f" {state} — yazılımcılar için fırsatlar var mı",
            f"{city_loc} emekli olmayı düşünüyorum, {state} — sağlık hizmetleri,"
            f" yaşam maliyeti ve genel yaşam kalitesi nasıl",
            f"{city_abl} banliyölerden {city} merkezine yoğun saatlerde"
            f" ulaşım süreleri ne kadar, {state}",
            f"{state_abl} gelen ziyaretçiler için {city_dat}"
            f" ulaşım ve konaklama rehberi",
        ]

    def templates_single_city_state_country(
        self, city: str, state: str, country: str
    ) -> list[str]:
        """Return query variants for city + state + country (Turkish)."""
        city_loc = self._inflect_text(city, "loc")
        city_dat = self._inflect_text(city, "dat")
        city_abl = self._inflect_text(city, "abl")
        state_loc = self._inflect_text(state, "loc")
        state_abl = self._inflect_text(state, "abl")
        country_loc = self._inflect_text(country, "loc")
        country_dat = self._inflect_text(country, "dat")
        country_abl = self._inflect_text(country, "abl")
        return [
            # -- short --
            f"{city}, {state}, {country}",
            f"{city} şehri, {state}, {country}",
            f"{city} posta kodu, {state}, {country}",
            f"{city} iklimi, {state}, {country}",
            f"{city} demografisi, {state}, {country}",
            f"{city_loc} faturalar, {state}, {country}",
            f"{city_loc} vergiler, {state}, {country}",
            f"{city} emniyet, {state}, {country}",
            # -- medium --
            f"{city_dat} posta kodu ve adres formatı, {state}, {country}",
            f"{state_loc} {city_abl} sonra {country_loc} nereye gidilir",
            f"{city_loc} iklim nasıl, {state}, {country}",
            f"{city_dat} kargo ve teslimat hizmetleri, {state}, {country}",
            f"{city_dat} konsolosluk bilgileri ve vize kuralları, {state}, {country}",
            f"{city_loc} oteller ve hosteller, {state}, {country}",
            f"{city}, {state}, {country}, ziyaret edilmeye değer mi",
            f"{city_loc} üniversiteler ve araştırma merkezleri, {state}, {country}",
            f"{city_loc} yapılacak şeyler, {state}, {country}",
            f"{city_loc} yaşam maliyeti dökümü, {state}, {country}",
            f"{city_loc} yöresel mutfak ve denenmesi gereken restoranlar, {state}, {country}",
            f"{city} tek başına seyahat edenler için ne kadar güvenli, {state}, {country}",
            f"{city_loc} halk kütüphaneleri ve kültürel mekanlar, {state}, {country}",
            f"{city} şehrini ziyaret etmek için en iyi mevsim, {state}, {country}",
            f"{city_loc} internet sağlayıcıları ve hızı, {state}, {country}",
            f"{city_loc} yerel ulaşım seçenekleri, {state}, {country}",
            f"{city_loc} emlak piyasası trendleri, {state}, {country}",
            f"{city_loc} gönüllülük ve hayır işleri, {state}, {country}",
            f"{city_loc} ithalat ve ihracat düzenlemeleri, {state}, {country}",
            f"{city_loc} ibadethaneler ve dini mekanlar, {state}, {country}",
            f"{city_loc} girişim kuluçka merkezleri ve teknoloji merkezleri, {state}, {country}",
            f"{city_loc} su kalitesi ve çevre raporu, {state}, {country}",
            f"{city_loc} hayvan sahiplendirme ve barınaklar, {state}, {country}",
            # -- long --
            f"{country_dat}, {state} bölgesine, {city} şehrine kargo göndermem"
            f" gerekiyor — posta kodu ve beklenen teslimat süresi nedir",
            f"{city_loc} kışın iki yatak odalı daire için ortalama faturalar"
            f" ne kadar, {state}, {country}",
            f"{city_dat} taşınırken konut bulmada en iyi yardımcı olan"
            f" emlak acenteleri hangileri, {state}, {country}",
            f"{city} şehrini {state_loc}, {country_loc} şirketimizin bölge ofisi"
            f" için potansiyel bir konum olarak değerlendiriyorum",
            f"{city_loc} öğrenciler için hangi staj ve çıraklık programları"
            f" mevcut, {state}, {country}",
            f"{city_loc} sağlık sistemini anlamak istiyorum, {state},"
            f" {country} — klinikler ve sigorta gurbetçiler için nasıl çalışıyor",
            f"{city_loc} serbest meslek kurmak için kurallar nelerdir,"
            f" {state}, {country}, ve hangi vergiler geçerli",
            f"{city_loc} küçük bir kafe veya restoran açmayı düşünüyorum,"
            f" {state}, {country} — hangi izinler gerekli",
            f"{state_abl} {country_abl} bölge özellikleri —"
            f" {city_dat} taşınırken bilinmesi gerekenler",
        ]

    # -- Two-city query templates ---------------------------------------------

    def templates_two_cities(self, city1: str, city2: str) -> list[str]:
        """Return query variants mentioning two cities (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1dat = self._inflect_text(city1, "dat")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        c2abl = self._inflect_text(city2, "abl")
        return [
            # -- short --
            f"{city1} ve {city2}",
            f"{city1} — {city2}",
            f"{city1} vs {city2}",
            f"{city1} veya {city2}",
            f"uçuşlar {city1} {city2}",
            f"mesafe {city1} {city2}",
            f"otobüs {c1abl} {c2dat}",
            f"tren {city1} {city2}",
            # -- medium --
            f"{c1abl} {c2dat} trenle veya otobüsle nasıl gidilir",
            f"kira fiyatları {c1loc} mı yoksa {c2loc} mi daha uygun",
            f"{city1} ile {city2} hava durumu karşılaştırması",
            f"aile tatili için {city1} mı yoksa {city2} mi daha iyi",
            f"{c1loc} ve {c2loc} restoranlar ve yemek kültürü",
            f"{c1abl} {c2dat} taşınmak — nelere dikkat etmeli",
            f"{city1} ve {city2} arasında güvenlik karşılaştırması",
            f"{c1loc} ve {c2loc} gece hayatı ve eğlence",
            f"{c1loc} ve {c2loc} üniversiteler karşılaştırması",
            f"{c1loc} ve {c2loc} sağlık hizmetleri kalitesi",
            f"{c1loc} ve {c2loc} market fiyatları",
            f"{city1} ile {city2} internet hızı karşılaştırması",
            f"{c1loc} ve {c2loc} hava kalitesi endeksi",
            f"{c1loc} ve {c2loc} evcil hayvan dostu mekanlar",
            f"{c1loc} ve {c2loc} ortak çalışma alanları",
            f"{city1} ve {city2} girişim ekosistemi",
            f"{c1loc} ve {c2loc} ulaşım süreleri",
            f"{c1loc} ve {c2loc} kültürel cazibe merkezleri",
            f"{c1loc} ve {c2loc} ortalama maaşlar",
            f"{c1loc} ve {c2loc} gurbetçi toplulukları",
            f"{c1loc} ve {c2loc} toplu taşıma kapsamı",
            f"{c1loc} ve {c2loc} yeşil alanlar ve parklar",
            # -- long --
            f"{c1abl} {c2dat} uçuş süresi ne kadar"
            f" ve hangi havayolları sefer düzenliyor",
            f"Hem {c1loc} hem {c2loc} iş teklifi aldım —"
            f" hangi şehirde maaşın satın alma gücü daha yüksek",
            f"{city1} ve {city2} toplu taşıma sistemlerini karşılaştırmak"
            f" istiyorum — arabasız yaşamak nerede daha kolay",
            f"{c1loc} ve {c2loc} hava kalitesi ve çevre koşullarını merak"
            f" ediyorum — hangisi daha temiz",
            f"Ailem {city1} ile {city2} arasında tartışıyor"
            f" — okullar nerede daha iyi ve çocuklar için daha güvenli",
            f"{c1abl} {c2dat} arabayla gidersem yol üzerinde görülmesi"
            f" gereken duraklar ve ilginç yerler nelerdir",
            f"{city1} ve {city2} emlak piyasaları nasıl karşılaştırılır"
            f" — nerede mülk almak için daha uygun",
            f"Serbest çalışanım ve {city1} ile {city2} arasında"
            f" seçim yapıyorum — nerede ortak çalışma alanları daha iyi",
            f"Hafta sonu kaçamağı planlıyorum — kısa iki günlük bir gezi"
            f" için {city1} mi yoksa {city2} mi keşfetmek daha iyi",
            f"{c1dat} veya {c2abl} kargo gönderimi"
            f" — hangi teslimat hizmetleri çalışıyor",
        ]

    def templates_two_cities_country(
        self, city1: str, country1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants for two cities, each with country (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1dat = self._inflect_text(city1, "dat")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        c2abl = self._inflect_text(city2, "abl")
        co1loc = self._inflect_text(country1, "loc")
        co1abl = self._inflect_text(country1, "abl")
        co2loc = self._inflect_text(country2, "loc")
        co2abl = self._inflect_text(country2, "abl")
        return [
            # -- short --
            f"{city1} ({country1}) ve {city2} ({country2})",
            f"{city1}, {country1} — {city2}, {country2}",
            f"vize {city1}, {country1}, ve {city2}, {country2}",
            f"para birimi {country1} ve {country2}, {city1} ve {city2}",
            f"uçuş {city1} {country1} — {city2} {country2}",
            f"iklim {city1} ({country1}) ve {city2} ({country2})",
            f"fiyatlar {city1} {country1} vs {city2} {country2}",
            f"kira {city1} ({country1}) vs {city2} ({country2})",
            # -- medium --
            f"{c1abl}, {country1}, {c2dat}, {country2}, seyahat için hangi belgeler gerekiyor",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, yaşam maliyeti karşılaştırması",
            f"kışın hangisi daha sıcak, {city1}, {country1}, mı yoksa {city2}, {country2}, mi",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, yaşam kalitesi",
            f"{city1} ({country1}) ve {city2} ({country2}) internet ve mobil kapsama",
            f"{country1} ile {country2} arasında gümrük kuralları, {city1} — {city2}",
            f"{co1loc} ({city1}) ve {co2loc} ({city2}) vergi farkları",
            f"{co1loc} ({city1}) ve {co2loc} ({city2}) sürüş kuralları",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, sağlık sistemleri",
            f"{country1} ({city1}) ve {country2} ({city2}) çalışma vizesi gereklilikleri",
            f"{country1} ve {country2} dijital göçebe vizeleri — {city1} vs {city2}",
            f"{co1loc} {city1} ile {co2loc} {city2} güvenlik karşılaştırması",
            f"{city1}, {country1}, ve {city2}, {country2}, saat dilimi farkı",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, IT maaş ortalamaları",
            f"{city1} ({country1}) ve {city2} ({country2}) uluslararası okullar",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, dışarıda yemek maliyeti",
            f"{city1} ({country1}) ve {city2} ({country2}) girişim ekosistemleri",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, gurbetçi ağları",
            f"{city1} ({country1}) vs {city2} ({country2}) toplu taşıma",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, emlak fiyatları",
            # -- long --
            f"{co1loc} {city1} ve {co2loc} {city2} şehirlerini bir gezide ziyaret"
            f" etmek istiyorum — rotayı nasıl planlayayım ve ayrı vize gerekir mi",
            f"Hem {c1loc} ({country1}) hem {c2loc} ({country2}) iş teklifi aldım —"
            f" aynı maaşla hangi şehirde satın alma gücü daha yüksek",
            f"{city1}, {country1}, ve {city2}, {country2}, toplu taşıma"
            f" sistemlerini karşılaştırmak istiyorum — arabasız nerede daha kolay",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, hava kalitesi ve"
            f" çevre koşullarıyla ilgileniyorum",
            f"İki çocuklu bir aile için hangi şehir daha iyi —"
            f" {city1}, {country1}, mi yoksa {city2}, {country2}, mi",
            f"{city1} ({country1}) ve {city2} ({country2}) şehirlerini şube"
            f" ofisi açmak için değerlendiriyorum",
            f"Staja gidiyorum — {co1loc} {city1} ile {co2loc} {city2} arasında seçim"
            f" yapıyorum, genç profesyoneller için koşullar nerede daha iyi",
            f"{country1} ve {country2} şehirlerini ziyaret etmeyi hayal ediyorum"
            f" — {c1abl} mı yoksa {c2abl} mı başlamalıyım",
            f"{c1loc}, {country1}, ve {c2loc}, {country2}, market ve günlük"
            f" harcama maliyetlerini karşılaştırmak istiyorum",
            f"{co1loc} ve {co2loc} çevre politikaları ve yeşil"
            f" girişimler nelerdir, {city1} ve {city2} karşılaştırması",
            f"{c1dat} veya {c2dat} kargo göndermek — {co1abl} ve"
            f" {co2abl} teslimat hizmetleri hangisi daha hızlı",
            f"Dil öğrenmek istiyorum — {c1loc} ({country1}) mı yoksa"
            f" {c2loc} ({country2}) mi daha iyi dil okulları var",
        ]

    def templates_two_cities_state(
        self, city1: str, state1: str, city2: str, state2: str
    ) -> list[str]:
        """Return query variants for two cities, each with state (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1dat = self._inflect_text(city1, "dat")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        c2abl = self._inflect_text(city2, "abl")
        return [
            # -- short --
            f"{city1} ({state1}) ve {city2} ({state2})",
            f"{city1}, {state1} — {city2}, {state2}",
            f"mesafe {city1} {state1} — {city2} {state2}",
            f"kira {city1} ({state1}) vs {city2} ({state2})",
            f"iklim {city1} {state1} ve {city2} {state2}",
            f"iş {city1}, {state1}, vs {city2}, {state2}",
            f"okullar {city1} ({state1}) {city2} ({state2})",
            f"suç {city1} ({state1}) vs {city2} ({state2})",
            # -- medium --
            f"{c1abl}, {state1}, {c2dat}, {state2}, toplu taşımayla nasıl gidilir",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, yaşam maliyeti karşılaştırması",
            f"taşınmak için {city1}, {state1}, veya {city2}, {state2}, arasında karar veriyorum",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, hava durumu karşılaştırması",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, okullar ve eğitim",
            f"{c1loc}, {state1}, ile {c2loc}, {state2}, iş piyasası",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, emlak vergileri",
            f"{city1} ({state1}) ve {city2} ({state2}) sağlık tesisleri",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, gece hayatı ve yemek",
            f"{city1} ({state1}) ile {city2} ({state2}) girişim ortamı",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, köpek dostu parklar",
            f"{city1} ({state1}) vs {city2} ({state2}) ulaşım süreleri ve trafik",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, spor salonu üyelikleri",
            f"{city1} ({state1}) ve {city2} ({state2}) yerel festivaller",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, market fiyatları",
            f"{city1} ({state1}) vs {city2} ({state2}) ortak çalışma alanları",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, ortalama maaşlar",
            f"{city1} ({state1}) ve {city2} ({state2}) müzeler ve kültürel mekanlar",
            f"{c1loc}, {state1}, vs {c2loc}, {state2}, halk kütüphaneleri",
            f"{city1} ({state1}) ile {city2} ({state2}) emekli yaşamı",
            f"{c1loc}, {state1}, ve {c2loc}, {state2}, gönüllü fırsatları",
            # -- long --
            f"Uygun fiyatlı konut arıyorum ve {city1}, {state1}, ile"
            f" {city2}, {state2}, emlak piyasalarını karşılaştırıyorum",
            f"Çocuk yetiştirmek için hangi şehir daha iyi — {state1} bölgesinde"
            f" {city1} mi yoksa {state2} bölgesinde {city2} mi — okul kalitesi, güvenlik",
            f"Eşimle iki şehir arasında karar veriyoruz — {state1} bölgesinde"
            f" {city1} veya {state2} bölgesinde {city2}, nerede daha sakin ve temiz",
            f"{c1loc}, {state1}, IT sektörünün gelişimiyle ilgileniyorum,"
            f" {c2loc}, {state2}, ile karşılaştırıldığında — nerede daha fazla iş var",
            f"{c1abl} ({state1}) {c2dat} ({state2}) yol gezisi"
            f" — yol boyunca görülmeye değer yerler neler",
            f"Üniversite öğrencileri için staj fırsatlarını karşılaştırıyorum,"
            f" {city1} ({state1}) ve {city2} ({state2})",
            f"{city1} ({state1}) ile {city2} ({state2}) arasında"
            f" her iki şehirden kolay ulaşılabilir bir yazlık arıyorum",
            f"Emeklilik için hangi şehir daha sakin —"
            f" {state1} bölgesinde {city1} mi yoksa {state2} bölgesinde {city2} mi",
            f"Çocuk bakımı ve anaokulu maliyetleri {c1loc}, {state1},"
            f" ile {c2loc}, {state2}, arasında nasıl karşılaştırılır",
            f"Uzaktan çalışıyorum ve {city1} ({state1}) ile {city2} ({state2})"
            f" arasında karar veriyorum — internet, kafeler ve topluluk",
            f"{c1dat} veya {c2abl} kargo göndermek istiyorum"
            f" — {state1} ve {state2} bölgelerinde teslimat süreleri ne kadar",
        ]

    def templates_two_cities_state_country(
        self, city1: str, state1: str, country1: str,
        city2: str, state2: str, country2: str,
    ) -> list[str]:
        """Return query variants for two cities, each with state + country (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        co1loc = self._inflect_text(country1, "loc")
        co1abl = self._inflect_text(country1, "abl")
        co2loc = self._inflect_text(country2, "loc")
        co2abl = self._inflect_text(country2, "abl")
        return [
            # -- short --
            f"{city1}, {state1}, {country1} — {city2}, {state2}, {country2}",
            f"uçuş {city1} ({state1}, {country1}) — {city2} ({state2}, {country2})",
            f"iklim {city1}, {state1}, {country1}, ve {city2}, {state2}, {country2}",
            f"fiyatlar {city1} ({country1}) vs {city2} ({country2})",
            f"yaşam {city1}, {country1}, ve {city2}, {country2}",
            f"maaşlar {city1} ({country1}) ve {city2} ({country2})",
            f"vize {city1} {country1} ve {city2} {country2}",
            f"kira {city1} ({state1}) vs {city2} ({state2})",
            # -- medium --
            f"{c1abl} ({state1}, {country1}) {c2dat} ({state2}, {country2}) nasıl gidilir",
            f"{c1loc}, {state1}, {country1}, ve {c2loc}, {state2}, {country2}, iklim karşılaştırması",
            f"{city1} ({state1}, {country1}) ile {city2} ({state2}, {country2}) uçak bileti",
            f"{c1loc}, {state1}, {country1}, ve {c2loc}, {state2}, {country2}, yaşam maliyeti",
            f"uzaktan çalışanlar için {city1} ({state1}, {country1}) mı {city2} ({state2}, {country2}) mi",
            f"{city1}, {state1}, {country1}, ve {city2}, {state2}, {country2}, vergi farkları",
            f"{city1} ({state1}, {country1}) vs {city2} ({state2}, {country2}) sağlık",
            f"{c1loc}, {state1}, {country1}, ve {c2loc}, {state2}, {country2}, güvenlik ve suç",
            f"{city1} ({state1}, {country1}) ile {city2} ({state2}, {country2}) okul sistemleri",
            f"{c1loc}, {state1}, {country1}, vs {c2loc}, {state2}, {country2}, emlak",
            f"{city1} ({country1}) ve {city2} ({country2}) kültürel yaşam",
            f"{c1loc}, {state1}, {country1}, vs {c2loc}, {state2}, {country2}, internet",
            f"{city1} ({state1}, {country1}) ve {city2} ({state2}, {country2}) gurbetçi blogları",
            f"{city1}, {country1}, vs {city2}, {country2}, yıl boyunca hava durumu",
            f"{co1loc} ({city1}) ve {co2loc} ({city2}) bankacılık seçenekleri",
            f"{c1loc}, {state1}, {country1}, vs {c2loc}, {state2}, {country2}, daire kiralama",
            f"{co1loc} ({city1}) ve {co2loc} ({city2}) ehliyet gereklilikleri",
            f"{city1} ({country1}) vs {city2} ({country2}) İngilizce yeterlilik",
            f"{city1} ({state1}) vs {city2} ({state2}) gece hayatı ve eğlence",
            # -- long --
            f"Yurt dışına taşınıyorum ve {state1} bölgesinde {city1}, {country1},"
            f" ile {state2} bölgesinde {city2}, {country2}, arasında karşılaştırma yapıyorum",
            f"İki çocuklu aile için hangi şehir daha iyi —"
            f" {city1}, {state1}, {country1}, mi yoksa {city2}, {state2}, {country2}, mi",
            f"{city1} ({state1}, {country1}) ile {city2} ({state2}, {country2})"
            f" arasında satın alma gücü farkını anlamak istiyorum",
            f"{city1} ({state1}, {country1}) ve {city2} ({state2}, {country2})"
            f" şube ofisi açma seçenekleri olarak değerlendiriyorum",
            f"Rota planlıyorum — önce {city1} ({state1}, {country1}),"
            f" sonra {city2} ({state2}, {country2}), en iyi ulaşım seçeneğini önerir misiniz",
            f"{city1} ({state1}, {country1}) ile {city2} ({state2}, {country2})"
            f" serbest meslek vergi yükünü karşılaştırın — nerede iş kurmak daha ucuz",
            f"{co1abl} {co2abl} vatandaşlar için vize — {city1} ({state1})"
            f" veya {city2} ({state2}) üzerinden hangisi daha kolay",
            f"Yazılım mühendisleri için iş piyasası nasıl görünüyor,"
            f" {city1} ({state1}, {country1}) ile {city2} ({state2}, {country2}) karşılaştırması",
            f"{c1loc}, {state1}, {country1}, ve {c2loc}, {state2}, {country2}, gurbetçi"
            f" yorumlarını topluyorum — blogumda deneyimleri paylaşıyorum",
            f"{city1}, {country1}, ile {city2}, {country2}, arasında gece hayatı,"
            f" sosyal yaşam ve arkadaşlık ortamını karşılaştırmak istiyorum",
            f"{co1abl} {city1} ({state1}) veya {co2abl} {city2} ({state2})"
            f" yabancı olarak banka hesabı açmak nerede daha kolay",
            f"{c1loc}, {state1}, {country1}, vs {c2loc}, {state2}, {country2},"
            f" hayvan sahiplendirme politikaları ve barınaklar",
            f"Franchising açmayı düşünüyorum — {city1} ({state1}, {country1})"
            f" mı yoksa {city2} ({state2}, {country2}) mi başlamak daha iyi",
        ]

    def templates_two_cities_one_country(
        self, city1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with country, city2 bare (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        c2abl = self._inflect_text(city2, "abl")
        co1loc = self._inflect_text(country1, "loc")
        co1abl = self._inflect_text(country1, "abl")
        return [
            # -- short --
            f"{city1}, {country1} ve {city2}",
            f"{city1}, {country1} — {city2}",
            f"{city1} ({country1}) vs {city2}",
            f"rota {city1}, {country1}, — {city2}",
            f"bilet {city1} ({country1}) — {city2}",
            f"mesafe {city1}, {country1}, — {city2}",
            f"{city1} ({country1}) ve {city2} — harita",
            f"uçuşlar {city1}, {country1}, {city2}",
            # -- medium --
            f"{c1abl}, {country1}, {c2dat} mesafe ve en iyi ulaşım yolu",
            f"{country1} gezisi planlıyorum, hem {city1} hem {city2} ziyaret etmek istiyorum",
            f"{city1}, {country1} ile {city2} arasında bilet arıyorum",
            f"hangisi daha büyük, {co1loc} {city1} mı yoksa {city2} mi",
            f"{city1}, {country1}, ve {city2} yemek ve gece hayatı karşılaştırması",
            f"yaşam maliyeti karşılaştırması {city1} ({country1}) vs {city2}",
            f"{city1}, {country1}, ile {city2} arasında hava durumu farkı",
            f"güvenlik karşılaştırması {co1loc} {city1} ile {city2}",
            f"{c1loc}, {country1}, ile {c2loc} emlak piyasası",
            f"gurbetçi hayatı {city1} ({country1}) vs {city2}",
            f"{c1loc}, {country1}, ile {c2loc} sağlık hizmetleri",
            f"{city1} ({country1}) ile {city2} okullar ve eğitim",
            f"{c1loc}, {country1}, vs {c2loc} toplu taşıma",
            f"{city1} ({country1}) ve {city2} ortalama maaşlar",
            f"{c1loc}, {country1}, vs {c2loc} gece hayatı seçenekleri",
            f"{city1} ({country1}) ile {city2} serbest meslek fırsatları",
            f"{c1loc}, {country1}, ve {c2loc} spor salonları",
            f"{city1} ({country1}) vs {city2} evcil hayvan dostu konaklama",
            f"{c1loc}, {country1}, ile {c2loc} ortak çalışma alanları",
            f"{city1} ({country1}) veya {c2abl} hafta sonu kaçamağı fikirleri",
            # -- long --
            f"Dijital göçebeyim ve {co1loc} {city1} ile {city2} şehrini"
            f" karşılaştırıyorum — nerede ortak çalışma alanları daha iyi ve maliyetler daha düşük",
            f"Küçük işletme açmak için {co1loc} {city1} ile {city2} arasında"
            f" seçim yapıyorum — nerede vergi koşulları daha iyi",
            f"Öğrenci için hangi şehir daha iyi — {co1loc} {city1} mi yoksa {city2} mi:"
            f" yurt maliyeti, yarı zamanlı iş, eğlence",
            f"Emlak bakıyorum ve {co1loc} {city1} piyasasını {city2} ile"
            f" karşılaştırıyorum — nereye yatırım yapmak daha akıllıca",
            f"{co1abl} {c2dat} turlar — {city1} üzerinden mi"
            f" yoksa doğrudan rota mı daha uygun",
            f"Teknoloji iş piyasası {c1loc}, {country1},"
            f" ile {c2loc} yazılım mühendisleri için nasıl karşılaştırılır",
            f"Emekliyim ve {co1loc} {city1} veya {city2} arasında"
            f" karar veriyorum — nerede sağlık daha iyi ve yaşam maliyeti düşük",
            f"{c1abl}, {country1}, {c2dat} araba yolculuğu"
            f" planlıyorum — sınır geçiş gereklilikleri ve manzaralı duraklar neler",
            f"{city1}, {country1}, ve {city2} arasında vergi oranları"
            f" ve iş düzenlemeleri farkı nedir",
            f"Yerel dili öğrenmek istiyorum — hangi şehirde dil"
            f" okulları daha iyi, {city1} ({country1}) mı yoksa {city2} mi",
            f"{co1abl} {c2dat} mobilya taşımak istiyorum"
            f" — hangi uluslararası nakliyeciler güvenilir",
            f"{c1loc} ({country1}) ile {c2loc} arasında sosyal yaşam"
            f" ve arkadaşlık ortamı genç profesyoneller için nasıl karşılaştırılır",
        ]

    def templates_two_cities_one_state(
        self, city1: str, state1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 bare (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        s1loc = self._inflect_text(state1, "loc")
        s1abl = self._inflect_text(state1, "abl")
        return [
            # -- short --
            f"{city1}, {state1} ve {city2}",
            f"{city1}, {state1} — {city2}",
            f"{city1} ({state1}) vs {city2}",
            f"yol {city1} ({state1}) — {city2}",
            f"tren {city1}, {state1}, — {city2}",
            f"mesafe {city1} ({state1}) — {city2}",
            f"{city1} ({state1}) ve {city2} harita",
            f"otobüs {city1} {state1} {city2}",
            # -- medium --
            f"{c1abl}, {state1}, {c2dat} mesafe ve arabayla kaç saat sürer",
            f"{s1loc} {city1} şehrini ziyaret edip {c2dat} de uğramak istiyorum",
            f"{city1} ({state1}) ile {city2} arasında otobüs seferleri var mı",
            f"hangi şehirde daha iyi okullar var, {s1loc} {city1} mı yoksa {city2} mi",
            f"{city1}, {state1}, ve {city2} emlak fiyatları karşılaştırması",
            f"{city1} ({state1}) ile {city2} hava durumu farkı",
            f"{c1loc}, {state1}, ile {c2loc} yaşam maliyeti",
            f"güvenlik karşılaştırması {s1loc} {city1} ile {city2}",
            f"{city1} ({state1}) ve {city2} sağlık seçenekleri",
            f"{c1loc}, {state1}, vs {c2loc} gece hayatı ve yemek",
            f"{city1} ({state1}) ile {city2} ortalama kira",
            f"{c1loc}, {state1}, ile {c2loc} iş piyasası",
            f"{city1} ({state1}) ve {city2} parklar ve rekreasyon",
            f"{c1loc}, {state1}, ile {c2loc} market fiyatları",
            f"{city1} ({state1}) ile {city2} ulaşım süreleri",
            f"{c1loc}, {state1}, ve {c2loc} evcil hayvan dostu konut",
            f"{city1} ({state1}) vs {city2} internet hızı",
            f"{c1loc}, {state1}, ve {c2loc} yaşlı bakım seçenekleri",
            f"{city1} ({state1}) ile {city2} kreş ve anaokulu",
            f"{c1loc}, {state1}, ve {c2loc} gönüllü fırsatları",
            # -- long --
            f"{s1loc} {city1} ile {city2} arasında seçim yapıyorum"
            f" — ilk kez ev alanlar için nerede konut piyasası daha uygun",
            f"Okul çağında çocuklu aileler için hangi şehir daha iyi —"
            f" {s1loc} {city1} mi yoksa {city2} mi, okullar ve güvenlik açısından",
            f"{c2loc} iş teklifi aldım ama şu anda {c1loc}, {state1}, yaşıyorum"
            f" — maaş ve yaşam kalitesi açısından taşınmaya değer mi",
            f"Çocuklarım üniversiteye başvuruyor — {city1} ({state1}) ve {city2}"
            f" şehirlerinde öğrenciler için fırsatları karşılaştırın",
            f"{city1} ({state1}) ile {city2} arasında, her iki şehirden"
            f" kolay ulaşılabilir bir yazlık arıyorum",
            f"Emeklilik için hangi şehir daha sakin —"
            f" {s1loc} {city1} mi yoksa {city2} mi, sağlık ve çevre açısından",
            f"{city1} ({state1}) ve {city2} şehirlerinde ekoloji ve gürültü"
            f" seviyelerini karşılaştırın — küçük çocuğum var ve sakin ortam önemli",
            f"{state1} bölgesinin şehirleri turu: {city1} ve {city2}"
            f" — bir hafta sonu gezisine neleri sığdırabilirsiniz",
            f"Uzaktan çalışıyorum ve {city1} ({state1}) ile {city2}"
            f" karşılaştırıyorum — internet ve kafe ortamı nerede daha iyi",
            f"{s1loc} {city1} ile {city2} arasında emlak vergileri nasıl"
            f" karşılaştırılır — yakında ev almayı planlıyorum",
            f"{s1abl} {c2dat} kargo göndermek istiyorum"
            f" — teslimat süresi ve en uygun hizmet hangisi",
            f"{c1loc} ({state1}) ve {c2loc} su kalitesi ve çevre raporlarını"
            f" karşılaştırıyorum — hangisi daha sağlıklı",
        ]

    def templates_two_cities_one_state_country(
        self, city1: str, state1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state + country, city2 bare (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        s1loc = self._inflect_text(state1, "loc")
        co1loc = self._inflect_text(country1, "loc")
        return [
            # -- short --
            f"{city1}, {state1}, {country1} ve {city2}",
            f"{city1}, {state1}, {country1} — {city2}",
            f"uçuş {city1} ({state1}, {country1}) — {city2}",
            f"bilet {city1}, {country1}, — {city2}",
            f"rota {city1} ({country1}) — {city2}",
            f"mesafe {city1}, {state1}, {country1}, — {city2}",
            f"{city1} ({state1}, {country1}) vs {city2}",
            f"harita {city1} {country1} {city2}",
            # -- medium --
            f"{c1abl} ({state1}, {country1}) {c2dat} mesafe ve nasıl gidilir",
            f"{city1}, {state1}, {country1}, ile {city2} arasında direkt uçuş var mı",
            f"{c1abl}, {state1}, {country1}, {c2dat} arabayla rota planlıyorum",
            f"{city1} ({state1}, {country1}) ile {city2} iklim karşılaştırması",
            f"{city1}, {state1}, {country1}, ile {city2} yaşam maliyeti",
            f"{city1} ({state1}, {country1}) ile {city2} hava durumu farkı",
            f"güvenlik değerlendirmesi {s1loc} {city1}, {country1}, ile {city2}",
            f"{c1loc}, {state1}, {country1}, ile {c2loc} sağlık kalitesi",
            f"{city1} ({state1}, {country1}) vs {city2} iş piyasası",
            f"{c1loc}, {state1}, {country1}, ile {c2loc} okullar ve eğitim",
            f"{city1} ({state1}, {country1}) ile {city2} ortalama kira",
            f"{c1loc}, {state1}, {country1}, vs {c2loc} toplu taşıma",
            f"gurbetçi hayatı {city1} ({state1}, {country1}) ile {city2}",
            f"{c1loc}, {country1}, vs {c2loc} gece hayatı ve yemek",
            f"{city1} ({state1}, {country1}) ile {city2} internet hızı",
            f"{c1loc}, {state1}, {country1}, ve {c2loc} ortak çalışma alanları",
            f"evcil hayvan dostu konut {city1} ({country1}) vs {city2}",
            f"{c1loc}, {state1}, {country1}, ile {c2loc} emlak yatırımı",
            f"serbest meslek vergi kuralları {city1} ({state1}, {country1}) vs {city2}",
            f"emeklilik seçenekleri {city1}, {country1}, ile {city2}",
            f"spor ve fitness ortamı {city1} ({state1}, {country1}) vs {city2}",
            # -- long --
            f"Küçük işletme açmak için {s1loc} {city1}, {country1}, ile"
            f" {city2} arasında seçim yapıyorum — nerede vergi koşulları daha iyi",
            f"Öğrenci için hangi şehir daha iyi — {city1} ({state1}, {country1})"
            f" mi yoksa {city2} mi: yurt maliyeti, yarı zamanlı iş, kampüs hayatı",
            f"{c1abl}, {state1}, {country1}, seyahat ediyorum ve {c2dat}"
            f" birkaç günlüğüne uğramak istiyorum — görülecek yerler neler",
            f"{c1loc} ({state1}, {country1}) yaşam ritmi {c2loc} ile nasıl"
            f" farklı — tempo, alışkanlıklar, zihniyet",
            f"{c1abl}, {state1}, {country1}, {c2dat} bagajlı uçak bileti"
            f" arıyorum gelecek ay — hangi havayolları bu rotada uçuyor",
            f"{c1loc} ({state1}, {country1}) ile {c2loc} market"
            f" maliyetleri ne kadar farklı — aylık bütçe planlamak istiyorum",
            f"IT profesyonelleri için iş piyasası analizi yapıyorum —"
            f" {city1} ({state1}, {country1}) ve {city2} fırsatlarını karşılaştırıyorum",
            f"{s1loc} {city1}, {country1}, şehrinden {c2dat} taşınmaya değer mi"
            f" — kariyer gelişimi, yaşam tarzı ve yaşam maliyetini tartıyorum",
            f"{co1loc} {city1} ve {city2} arasında gönüllü ve hayır"
            f" kuruluşları hangileri faaliyet gösteriyor",
            f"Sosyal etkinlikler ve flört ortamını {c1loc} ({state1},"
            f" {country1}) ile {c2loc} karşılaştırıyorum",
            f"Uygun fiyatlı çocuk bakımı seçenekleri hangi şehirde daha fazla —"
            f" {s1loc} {city1} ({country1}) mi yoksa {city2} mi",
        ]

    def templates_two_cities_state_and_country(
        self, city1: str, state1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 with country (Turkish)."""
        c1loc = self._inflect_text(city1, "loc")
        c1abl = self._inflect_text(city1, "abl")
        c2loc = self._inflect_text(city2, "loc")
        c2dat = self._inflect_text(city2, "dat")
        s1loc = self._inflect_text(state1, "loc")
        co2loc = self._inflect_text(country2, "loc")
        co2abl = self._inflect_text(country2, "abl")
        return [
            # -- short --
            f"{city1} ({state1}) ve {city2} ({country2})",
            f"{city1}, {state1} — {city2}, {country2}",
            f"uçuşlar {city1} ({state1}) — {city2} ({country2})",
            f"kira {city1} ({state1}) vs {city2} ({country2})",
            f"hava durumu {city1} {state1} ve {city2} {country2}",
            f"iş {city1} ({state1}) vs {city2} ({country2})",
            f"mesafe {city1} {state1} — {city2} {country2}",
            f"vize {city1} — {city2}, {country2}",
            # -- medium --
            f"{c1abl}, {state1}, {c2dat}, {country2}, nasıl gidilir",
            f"{c1loc}, {state1}, ve {c2loc}, {country2}, kira fiyatları karşılaştırması",
            f"{city1} ({state1}) ile {city2}, {country2}, bilet arıyorum",
            f"{c1loc}, {state1}, veya {c2loc}, {country2}, yaşam kalitesi hangisinde daha iyi",
            f"{c1loc}, {state1}, ile {c2loc}, {country2}, iş fırsatları",
            f"{city1}, {state1}, ile {city2}, {country2}, iklim farkı",
            f"güvenlik karşılaştırması {s1loc} {city1} ile {co2loc} {city2}",
            f"{city1} ({state1}) ve {city2} ({country2}) sağlık sistemleri",
            f"{c1loc}, {state1}, vs {c2loc}, {country2}, okullar ve üniversiteler",
            f"{city1} ({state1}) ile {city2} ({country2}) toplu taşıma",
            f"{c1loc}, {state1}, ile {c2loc}, {country2}, emlak piyasası",
            f"{city1} ({state1}) ve {city2} ({country2}) market fiyatları",
            f"{c1loc}, {state1}, vs {c2loc}, {country2}, internet hızı ve bağlantı",
            f"{city1} ({state1}) ile {city2} ({country2}) ortalama maaşlar",
            f"{c1loc}, {state1}, ve {c2loc}, {country2}, gurbetçi toplulukları",
            f"{city1} ({state1}) vs {city2} ({country2}) gece hayatı",
            f"{c1loc}, {state1}, ile {c2loc}, {country2}, ortak çalışma alanları",
            f"{city1} ({state1}) ve {city2} ({country2}) evcil hayvan dostu konaklama",
            f"{c1loc}, {state1}, ile {c2loc}, {country2}, girişim ortamı",
            f"{city1} ({state1}) vs {city2} ({country2}) bankacılık ve finans",
            f"{c1loc}, {state1}, ve {c2loc}, {country2}, kültürel etkinlikler",
            f"{city1} ({state1}) vs {city2} ({country2}) spor ve fitness",
            # -- long --
            f"{s1loc} {city1} ile {co2loc} {city2} arasında seçim yapıyorum"
            f" — genç profesyoneller için konut piyasası nerede daha uygun",
            f"Aile için hangi şehir daha iyi — {city1} ({state1})"
            f" mi yoksa {city2} ({country2}) mi, okullar, güvenlik ve parklar açısından",
            f"Arkadaşlarım {c2dat} ({country2}) taşınmamı istiyor ama {c1loc},"
            f" {state1}, alıştım — her iki seçeneğin artıları ve eksileri neler",
            f"Hangi havayolları {c1abl}, {state1},"
            f" {c2dat}, {country2}, uçuyor ve bilet indirimleri ne sıklıkla oluyor",
            f"Teknoloji çalışanları için iş piyasası analizi yapıyorum —"
            f" {city1} ({state1}) ve {city2} ({country2}) iş ilanlarını karşılaştırıyorum",
            f"Emekliler için hangi şehir daha rahat — {s1loc} {city1}"
            f" mi yoksa {co2loc} {city2} mi, sağlık ve yaşam maliyeti açısından",
            f"{s1loc} {city1} şehrine hayranlık duyuyorum ama"
            f" {co2abl} şehirlerle, özellikle {city2} ile karşılaştırmak istiyorum",
            f"{c1abl}, {state1}, {c2dat}, {country2},"
            f" arabayla yolculuk — süre ne kadar ve manzaralı duraklar neler",
            f"Serbest meslek vergi yükünü {city1} ({state1})"
            f" ile {city2} ({country2}) karşılaştırmak istiyorum — nerede iş kurmak daha ucuz",
            f"İki haftalık tatil planlıyorum — önce {s1loc} {city1},"
            f" sonra {co2loc} {city2}, aralarında en iyi ulaşım nedir",
        ]
    
    def city_description(
        self,
        city: str,
        country: str,
        admin1_name: str,
        other_names: list[str],
    ) -> str:
        """Build a short Turkish description of a city."""
        text = f"{city} şehri, {country} ülkesi, {admin1_name} ili"
        if other_names:
            text += f". Ayrıca şu adla da bilinir: {', '.join(other_names)}"
        return text


# Auto-register on import
register_language(TurkishLanguage())
