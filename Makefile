DOCKER_IMAGE := geosearch
DOCKER_PORT  := 8000

.PHONY: install preprocess build-index data run docker-build docker-run all

install:
	uv sync

preprocess:
	uv run python -m src.data.preprocess_cities.run

build-index:
	uv run python -m src.data.build_index

## Запустить preprocess + build-index последовательно
data: preprocess build-index

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
