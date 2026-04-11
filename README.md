# GeoSearch

Сервис полнотекстового гео-поиска по текстовому запросу на русском, английском и турецком языках.  
Поиск реализован на BM25 (character n-grams) поверх данных [GeoNames](https://www.geonames.org/), API — FastAPI.

---

## Архитектура

```text
geosearch/
├── data/
│   ├── raw/                         # Исходные данные GeoNames
│   │   ├── countries/               # RU.txt, US.txt, TR.txt
│   │   ├── alternateNamesV2.txt
│   │   └── admin1CodesASCII.txt
│   └── processed/
│       ├── cities.parquet           # Результат preprocessing городов
│       ├── regions.parquet          # Результат preprocessing регионов
│       └── bm25_index.pkl           # Сериализованный BM25 индекс
│
├── src/
│   ├── data/
│   │   ├── config.py                # Глобальные константы и пути
│   │   ├── translation.py           # LLM-перевод через DeepSeek
│   │   ├── build_index.py           # CLI: сборка и сохранение BM25 индекса
│   │   ├── preprocess_cities/
│   │   │   ├── loader.py            # Загрузка и структурирование сырых данных
│   │   │   ├── pipeline.py          # Заполнение пропущенных переводов
│   │   │   └── run.py               # CLI: запуск preprocessing городов
│   │   └── preprocess_regions/
│   │       ├── utils.py             # scan_admin1_codes, per-country процессоры и таблицы диспетчеризации
│   │       ├── pipeline.py          # Сбор кандидатов, LLM-перевод, диспетчеризация по странам
│   │       └── run.py               # CLI: запуск preprocessing регионов
│   │
│   ├── models/
│   │   └── bm25_index.py            # GeoSearchIndex — from_parquet / load / save / search
│   │
│   └── api/
│       ├── schemas.py               # Pydantic-модели ответов
│       ├── app.py                   # FastAPI app factory + lifespan
│       └── routers/
│           └── v1/
│               └── search.py        # GET /v1/search
│
├── Dockerfile
├── Makefile
└── main.py                          # Точка входа uvicorn
```

---

## Что реализовано

### Preprocessing городов (`src/data/preprocess_cities/`)

Преобразует сырые файлы GeoNames в структурированный Parquet-датасет.

**Шаги пайплайна:**

1. **Загрузка** — читаются файлы стран (`RU.txt`, `US.txt`, `TR.txt`), фильтруются только населённые пункты (`feature_class = P`, `population > 0`).
2. **Нормализация имён** — подключается `alternateNamesV2.txt`; для русского языка отфильтровываются транслитерированные записи (содержащие латиницу, кроме римских цифр).
3. **Дедупликация** — двухуровневая агрегация объединяет дублирующиеся записи одного города в одну строку.
4. **Заполнение пропусков** — города без имени на каком-либо языке получают перевод:
   - кириллические языки (`ru`, `uk`, …) — перевод через LLM (DeepSeek);
   - остальные — фолбэк на английское имя.
5. **Сохранение** — `data/processed/cities.parquet`.

### Preprocessing регионов (`src/data/preprocess_regions/`)

Переводит названия административных регионов первого уровня (admin1) из `admin1CodesASCII.txt` на все целевые языки.

**Стратегия перевода** изолирована по странам в `utils.py`:

| Страна | `en` | `ru` | `tr` |
|--------|------|------|------|
| RU | `ascii_name` | LLM | LLM |
| TR | `ascii_name` | LLM | колонка `name` (нативное) |
| US | `ascii_name` | LLM | `ascii_name` |
| fallback | `ascii_name` | `ascii_name` | `ascii_name` |

`pipeline.py` не содержит логики конкретных стран — он опирается на два словаря из `utils.py`: `PROCESSORS` (как строить строки) и `TRANSLATE_LANGS` (какие языки отправлять в LLM).

**Поддержка новых стран:** страны без записи в `PROCESSORS` автоматически попадают в `process_fallback`. Для полной поддержки достаточно написать функцию `process_XX` и добавить её в оба словаря в `utils.py`.

**Сохранение** — `data/processed/regions.parquet` (схема: `admin1_code`, `country_code`, `language`, `name`).

### Поисковый индекс (`src/models/bm25_index.py`)

`GeoSearchIndex` строит BM25 индекс над всеми альтернативными именами городов:

- Токенизация: **character 3-grams** — устойчива к опечаткам и разным языкам.
- Каждый документ в корпусе — одно уникальное имя города.
- После BM25-скоринга результаты **переранжируются по численности населения**, дубликаты удаляются.
- Индекс строится один раз (`build_index.py`) и сохраняется в `bm25_index.pkl` — приложение загружает его из pickle, не пересобирая при каждом старте.

### REST API (`src/api/`)

| Метод | Путь | Описание |
| --- | --- | --- |
| `GET` | `/v1/search` | Поиск городов по тексту |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

**Параметры `/v1/search`:**

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `q` | `string` | обязательный | Поисковый запрос |
| `top_k` | `int` | `20` | Количество результатов (1–100) |

**Пример ответа:**

```json
{
  "query": "Москва",
  "total": 1,
  "results": [
    {
      "geoname_id": 524901,
      "ascii_name": "Moscow",
      "country_code": "RU",
      "admin1_code": "48",
      "latitude": 55.75222,
      "longitude": 37.61556,
      "population": 10381222
    }
  ]
}
```

---

## Установка

Требуется Python 3.12+ и [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd geosearch
```

Создайте `.env` в корне проекта:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

---

## Запуск локально

### Полный цикл с нуля

```bash
make all        # uv sync + preprocess + build-index
make run        # запуск сервера
```

### Пошагово

```bash
make install           # установить зависимости (uv sync)
make preprocess        # собрать cities.parquet из сырых данных GeoNames
make preprocess-regions # собрать regions.parquet из admin1CodesASCII.txt
make build-index       # собрать bm25_index.pkl
make run               # запустить сервер на :8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Запуск в Docker

Данные подготавливаются на хосте и монтируются в контейнер как volume — образ содержит только код и зависимости.

```bash
# 1. Подготовить данные (один раз, на хосте)
make install
make data

# 2. Собрать образ
make docker-build

# 3. Запустить контейнер
make docker-run
```

Или вручную:

```bash
docker build -t geosearch .
docker run --rm -p 8000:8000 -v $(PWD)/data/processed:/app/data/processed geosearch
```

---

## Примеры запросов

```bash
# По-русски
curl "http://localhost:8000/v1/search?q=Москва&top_k=5"

# По-английски
curl "http://localhost:8000/v1/search?q=New+York&top_k=10"

# По-турецки
curl "http://localhost:8000/v1/search?q=Istanbul"
```
