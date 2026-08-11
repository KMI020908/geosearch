---
title: GeoSearch
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Multilingual toponym recognition and matching over GeoNames
models:
  - mki0809/geosearch-ner
  - mki0809/geosearch-reranker
datasets:
  - mki0809/geosearch-index
  - mki0809/geosearch-queries
---

# GeoSearch

Multilingual toponym recognition and matching. Give it free text in Russian,
English, Turkish or Chinese; it finds the place names and returns ranked
[GeoNames](https://www.geonames.org/) entries.

```
text
  -> GLiNER NER (fine-tuned)
  -> char-n-gram BM25 retrieval
  -> cross-encoder reranker
  -> ranked places
```

## What the demo shows

Results are rendered **twice**: with the reranker, and in plain retriever order.
Both rank the same candidate set — retrieval ends by sorting on BM25 score with
population as the tiebreak and cutting to top-*k*, and the reranker only permutes
those survivors — so the difference between the two tables is exactly what the
reranker contributes. On a homonym query (`Springfield, Illinois`) that is the
whole disambiguation story in one screen.

The `莫斯科新闻` example is worth trying: the fine-tuned model segments Chinese
per ideograph, so it can find `莫斯科` inside a longer run of Han characters —
something the stock whitespace splitter cannot express at all.

## No database

Nowhere in this project, in fact. The engine serves prebuilt artifacts pulled
from [`geosearch-index`](https://huggingface.co/datasets/mki0809/geosearch-index):
a pickle-free BM25 index, a places table, and the reranker's candidate
documents. They are compiled from a Parquet staging corpus on the authoring
machine; nothing at request time reads anything but these files.

## Cold start

**The first request after the Space wakes takes a few minutes.** About 1.3 GB
has to download (1.15 GB of GLiNER weights, ~90 MB index, ~45 MB of tables) and
then load, and a free Space has no persistent storage, so every cold boot pays
it again. The UI comes up immediately and waits, rather than leaving the port
unbound.

Subsequent queries are fast: retrieval over 1.19M name strings takes about a
millisecond.

## Attribution

Place data from [GeoNames](https://www.geonames.org/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modified: filtered to
populated places (`feature_class='P'`) in RU/US/TR/CN, name variants restricted
to ru/en/tr/zh, some feature codes dropped, and names grouped per place into
document strings.

Built from source revision `{{GIT_SHA}}`.
