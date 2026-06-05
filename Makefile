DOCKER_IMAGE := geosearch
DOCKER_PORT  := 8000

.PHONY: install preprocess preprocess-regions build-index collect-queries generate-datasets retriever-data train-retriever run docker-build docker-run all

install:
	uv sync

preprocess:
	uv run python -m src.data.preprocess_cities.run

preprocess-regions:
	uv run python -m src.data.preprocess_regions.run

## Собрать BM25 индекс
build-index:
	uv run python -m src.data.build_index

## Сгенерировать шаблонные запросы по всем языкам → data/processed/template_queries/
collect-queries:
	uv run python -m src.data.collect_template_queries.run

## Разбить template_queries на train/val → data/processed/retriever_datasets/
generate-datasets:
	uv run python -m src.data.generate_retriever_datasets.run

## Полный пайплайн сборки датасета для ретривера
retriever-data: collect-queries generate-datasets

## Дообучить biencoder
train-biencoder:
	uv run python -m src.models.train_biencoder

run:
	uv run python main.py

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm \
		-p $(DOCKER_PORT):8000 \
		-v $(PWD)/data/processed:/app/data/processed \
		$(DOCKER_IMAGE)

## Полный цикл с нуля: зависимости → данные
all: install data
