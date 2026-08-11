"""Render the README that ships with each Hub repo.

Every number on a card comes from a metrics file this project wrote, never from
a literal typed here. That is the whole design rule: a card is the public claim
about an artifact, and a claim no file supports is exactly the kind of thing
that drifts one re-train after it stops being true.

Where a metrics file is absent, the card says so rather than omitting the
section — "not measured" is information; a missing table reads as an oversight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import Settings

# GeoNames is CC-BY 4.0, which obliges three things on every derivative: credit,
# a link to the licence, and an indication of what was changed. The last is the
# one that is easy to forget, so it is spelled out rather than gestured at.
GEONAMES_ATTRIBUTION = """\
## Data licence and attribution

Derived from [GeoNames](https://www.geonames.org/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Modifications made to the source data:**

- filtered to populated places only (`feature_class = 'P'`)
- restricted to {countries}
- restricted to name variants in {languages}
- dropped feature codes {excluded}
- name variants grouped per place and joined into single document strings
"""


def _scope(settings: Settings) -> dict[str, str]:
    return {
        "countries": ", ".join(settings.countries),
        "languages": ", ".join(settings.languages),
        "excluded": ", ".join(settings.excluded_feature_codes),
    }


def _read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _table(header: list[str], rows: list[tuple[str, ...]]) -> str:
    """Render a markdown table from rows, so long cells need not be long lines."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(cells) + " |" for cells in rows)
    return "\n".join(out)


def _score_table(scores: dict[str, Any], prefix: str | None = None) -> str:
    """Render P/R/F1 rows, keeping the support counts beside every rate.

    The support is not decoration. The val split is ~47 queries, so a bucket
    like `label:STATE` rests on a handful of spans and moves by tenths of F1 on
    one of them — a bare 0.83 invites a confidence the sample cannot carry.
    """
    rows = [
        "| bucket | precision | recall | F1 | gold | predicted |",
        "|---|---|---|---|---|---|",
    ]
    for bucket, s in scores.items():
        if prefix is not None and not bucket.startswith(prefix):
            continue
        label = bucket.split(":", 1)[-1] if prefix else bucket
        rows.append(
            f"| {label} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['gold']} | {s['predicted']} |"
        )
    return "\n".join(rows) if len(rows) > 2 else "_No buckets in this group._"


def render_ner_card(
    meta: dict[str, Any], metrics: dict[str, Any] | None, settings: Settings
) -> str:
    """Model card for the fine-tuned GLiNER checkpoint."""
    hp = meta.get("hyperparameters", {})
    scope = _scope(settings)
    base = meta.get("base_model", "unknown")
    base_url = f"https://huggingface.co/{meta.get('base_model', '')}"

    if metrics and (scores := metrics.get("scores")):
        overall = scores.get("overall", {})
        headline = (
            f"**Span F1 {overall.get('f1', 0):.3f}** "
            f"(P {overall.get('precision', 0):.3f} / R {overall.get('recall', 0):.3f}) "
            f"on {metrics.get('queries', '?')} held-out queries "
            f"at threshold {metrics.get('threshold', '?')}."
        )
        overall_block = _score_table({"overall": overall})
        by_label = _score_table(scores, prefix="label:")
        by_language = _score_table(scores, prefix="lang:")
        baseline = metrics.get("baseline")
        if baseline and (b := baseline.get("scores", {}).get("overall")):
            baseline_block = (
                f"Zero-shot `{baseline.get('model')}`, measured with the **same** "
                f"word splitter (otherwise the Chinese delta would be an artefact "
                f"of segmentation rather than of fine-tuning):\n\n"
                + _score_table({"overall": b})
            )
        else:
            baseline_block = "_No baseline in the metrics file._"
    else:
        headline = "_Metrics have not been generated — run `make ner-eval`._"
        overall_block = by_label = by_language = baseline_block = (
            "_Not measured. Run `make ner-eval`._"
        )

    return f"""\
---
license: cc-by-4.0
language: [ru, en, tr, zh]
library_name: gliner
pipeline_tag: token-classification
tags: [gliner, ner, toponym, geonames, multilingual]
base_model: {meta.get("base_model", "unknown")}
---

# geosearch-ner

Multilingual toponym NER, fine-tuned from [`{base}`]({base_url}).
Labels: {", ".join(f"`{x}`" for x in meta.get("labels", []))}.

{headline}

This is the first stage of a multilingual toponym search pipeline:

```
text
  -> GLiNER (this model)
  -> char-n-gram BM25 retrieval
  -> cross-encoder reranker
  -> ranked GeoNames places
```

## Serving it: the word-splitter contract

**This checkpoint requires a per-ideograph word splitter, and serving it without
one loses every Chinese span with no error of any kind.**

GLiNER classifies *token spans*, and its default whitespace splitter treats a
run of Han characters as one token — so `莫斯科` inside `莫斯科新闻` is not a
span the model can even express, let alone predict. This model was trained with
one token per Han ideograph (`src/ner/tokenizer.py::CjkAwareSplitter`), which
`gliner_config.json` cannot record: `words_splitter_type` names only GLiNER's
built-in kinds.

So the requirement is recorded in **`ner_meta.json`**, shipped beside the
weights in this repo:

```json
{{"words_splitter": "{meta.get("words_splitter", "?")}"}}
```

The serving code reads that file and refuses to start when the configured
splitter disagrees with it. If you load this model yourself, re-attach an
equivalent splitter — measured effect of getting it wrong: 56/56 Chinese spans
found versus 0.

## Decision threshold

Served at **{meta.get("eval_threshold", "?")}**, below GLiNER's own 0.5, and
that is a property of *serving* rather than of the model: retrieval is
recall-hungry, because a city span never extracted can never be retrieved,
while a spurious span only adds a candidate the reranker can demote. The value
is derived from the sweep in `make ner-eval --sweep`, not asserted.

## Results

### Overall

{overall_block}

### By label

{by_label}

### By language

{by_language}

### Versus the zero-shot baseline

{baseline_block}

Metrics are micro-averaged over **spans**, not macro over queries: a query
naming three cities feeds three names into retrieval, so each one is a unit of
work the pipeline either gets right or does not.

## Training

Selected by **validation span F1**, not `eval_loss` — the checkpoint saved is
whichever epoch scored best when called the way the engine calls it
(`predict_entities` on raw text).

| | |
|---|---|
| best epoch | {meta.get("best_epoch", "?")} |
| best val span F1 | {meta.get("best_val_span_f1", "?")} |
| epochs | {hp.get("epochs", "?")} |
| batch size | {hp.get("batch_size", "?")} |
| learning rate | {hp.get("learning_rate", "?")} |
| others LR | {hp.get("others_lr", "?")} |
| weight decay | {hp.get("weight_decay", "?")} |
| warmup ratio | {hp.get("warmup_ratio", "?")} |
| seed | {hp.get("seed", "?")} |
| trained at | {meta.get("trained_at", "?")} |

Trained on synthetic queries generated for this project — see
[`{settings.hf.datasets_repo}`](https://huggingface.co/datasets/{settings.hf.datasets_repo}).

{GEONAMES_ATTRIBUTION.format(**scope)}

The fine-tune inherits any licence conditions of its base model; check
[`{base}`]({base_url})
before commercial use.
"""


def render_reranker_card(
    meta: dict[str, Any], metrics: dict[str, Any] | None, settings: Settings
) -> str:
    """Model card for the fine-tuned cross-encoder reranker."""
    from src.rerank.features import NAME_SEPARATOR

    base_model = meta.get("base_model", "unknown")
    base_url = f"https://huggingface.co/{base_model}"

    if metrics:
        rows = ["| system | " + " | ".join(metrics.get("columns", [])) + " |"]
        rows.append("|---" * (1 + len(metrics.get("columns", []))) + "|")
        for name, values in metrics.get("rows", {}).items():
            rows.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        results = "\n".join(rows)
    else:
        results = "_Not measured. Run `make rerank-eval`._"

    hp = meta.get("hyperparameters", {})
    hp_rows = "\n".join(
        f"- {name}: {hp.get(key, '?')}"
        for name, key in (
            ("epochs", "epochs"),
            ("batch size", "batch_size"),
            ("learning rate", "learning_rate"),
            ("warmup ratio", "warmup_ratio"),
            ("weight decay", "weight_decay"),
            ("max length", "max_length"),
            ("pos_weight (BCE)", "pos_weight"),
            ("seed", "seed"),
        )
    )

    return f"""\
---
license: cc-by-4.0
language: [ru, en, tr, zh]
library_name: sentence-transformers
base_model: {base_model}
tags: [cross-encoder, sentence-transformers, reranker, geonames, toponym]
---

# geosearch-reranker

Cross-encoder reranker fine-tuned from [`{base_model}`]({base_url}), the last
stage of a multilingual toponym search pipeline:

```
text
  -> GLiNER NER
  -> char-n-gram BM25 retrieval
  -> cross-encoder reranker (this model)
  -> ranked GeoNames places
```

It **reorders** a fixed candidate set; it does not choose one. Retrieval ends by
ranking on BM25 score with population as the tiebreak and cutting to top-k, and
the model only permutes those survivors.

## Results

{results}

## Input format

The model scores raw `(query_text, document)` text pairs — no hand-engineered
features. `query_text` is `" ".join(entities)`, the flat NER span texts joined
in occurrence order — the exact same string BM25 retrieval scores against, not
the typed city/country/admin1 split (that split is a display-only field on the
API response). `document` is three lines:

```
<name> {NAME_SEPARATOR.strip()} <name> {NAME_SEPARATOR.strip()} <name>
<country in English>
<admin1 region in English>
```

The `{NAME_SEPARATOR!r}` separator is load-bearing. Joined by a space instead,
a place's spellings collapse into one pseudo-name a span naming it once would
only partially overlap with. Separated, each spelling stands on its own.

A cross-encoder attends jointly over both texts in a single forward pass, so —
unlike the CatBoost model this project used before — there is no need for
hand-computed span/document overlap features: the previous model's bag-of-words
text columns could not see the *intersection* between a query span and the
document, which is exactly what a cross-encoder's cross-attention computes
directly.

## Training

Fine-tuned pointwise with `BinaryCrossEntropyLoss` on the mined
`(query_text, document, label)` pairs (:mod:`src.rerank.dataset`) — one row per
retrieved candidate, gold geonameid(s) labelled positive, everything else
negative. `top_k=50` retrieval mines roughly one positive per query against up
to 50 negatives, so the loss is given a `pos_weight` computed from the train
split's own class ratio.

{hp_rows}
- best epoch: {meta.get("best_epoch", "?")}
- best test P@1: {meta.get("best_p_at_1", "?")}
- trained at: {meta.get("trained_at", "?")}

Model selection is by P@1 on the held-out mined pairs (grouped by query),
scored after every epoch and saved on improvement — not by the pointwise
`eval_loss`, which doesn't reflect whether the top-ranked candidate within a
query's group is the gold one. Split by **query**, not by place: a query is the
ranking group, so splitting on geonameid would tear one query's candidates
across train and test — leaking the query and leaving positive-less test groups.

## Training provenance

- `use_gold_entities`: **{meta.get("use_gold_entities", "unknown")}**

That flag has to be `false` for a servable model. `true` trains on the query
dataset's gold spans instead of mined NER output — a useful ablation ("how good
would this be if NER were perfect?") that breaks train/serve parity by design,
because online the reranker always receives GLiNER spans. A model trained with
it set must not be served, and this line is on the card so that cannot happen
silently.

## Known gap

`admin1_name` exists in English only (`admin1CodesASCII.txt`), so the
document's admin1 line rarely matches a ru/zh query lexically. Regions are
`feature_class='A'` and the ETL loads only `'P'`, so no localised region names
were ever ingested.

{GEONAMES_ATTRIBUTION.format(**_scope(settings))}

The fine-tune inherits any licence conditions of its base model; check
[`{base_model}`]({base_url}) before commercial use.
"""


def render_index_card(manifest: Any, settings: Settings) -> str:
    """Dataset card for the serving artifacts."""
    files = _table(
        ["file", "what it is"],
        [
            (
                "`bm25_index.npz`",
                f"character-n-gram BM25 index over {manifest.n_documents:,} name "
                "strings, plus the geonameid groups each name resolves to",
            ),
            (
                "`places.parquet`",
                f"{manifest.n_places:,} places — the fields a result needs "
                "(name, country, population, feature code, coordinates)",
            ),
            (
                "`descriptions.parquet`",
                f"{manifest.n_descriptions:,} candidate documents for the reranker",
            ),
            ("`manifest.json`", "what this build is, and what it was built from"),
        ],
    )
    return f"""\
---
license: cc-by-4.0
language: [ru, en, tr, zh]
tags: [geonames, bm25, retrieval, search-index]
---

# geosearch-index

Prebuilt **serving artifacts** for a multilingual toponym search engine: enough
to answer queries with no database of any kind.

{files}

**No pickle.** The index is plain numpy arrays loaded with
`allow_pickle=False`. An earlier format was a pickled object graph, which is
arbitrary code execution on load — acceptable for a file you just wrote
yourself, not for one fetched from a public repo.

## These four files are one set

`manifest.json` carries a `build_id`, and the two Parquet files are stamped with
it. That is checked at load, because the failure it prevents is silent: a places
table missing a geonameid the index can retrieve does not raise, it just drops
that place from every result it should have won.

```
build_id  {manifest.build_id}
built_at  {manifest.built_at}
```

## Scope

| | |
|---|---|
| countries | {", ".join(manifest.countries)} |
| languages | {", ".join(manifest.languages)} |
| excluded feature codes | {", ".join(manifest.excluded_feature_codes)} |

Scope is recorded because it decides *what is in* the corpus, not merely how it
is stored — an index built for two countries and served under a config naming
four cannot return the missing two, and nothing downstream would notice.

## Retrieval is name-level, not place-level

One BM25 document per unique name string, backed by every geonameid carrying it.
So "Moscow" is a single document covering the capital and every hamlet sharing
the spelling, and they all inherit the same retriever score — homonyms tie at
retrieval time by construction. Disambiguation is deferred to the population
tiebreak and the reranker, not solved by the retriever.

IDF is pinned to 1, making the score pure term-frequency saturation over shared
character n-grams: deliberately tolerant of transliteration and fuzzy spelling
across {", ".join(manifest.languages)}, at the cost of rare-term weighting.

## Checksums

```
{chr(10).join(f"{name}  {digest}" for name, digest in manifest.sha256.items())}
```

{GEONAMES_ATTRIBUTION.format(**_scope(settings))}
"""


def render_dataset_card(stats: dict[str, Any], settings: Settings) -> str:
    """Dataset card for the synthetic query corpus and derived training sets."""
    configs = _table(
        ["config", "rows", "what it is"],
        [
            (
                "`queries`",
                str(stats.get("queries", "?")),
                "the generated queries, with gold geonameids and spans",
            ),
            (
                "`plan`",
                str(stats.get("plan", "?")),
                "one row per intended generation call — provenance for `queries`",
            ),
            (
                "`rerank`",
                f"{stats.get('rerank_train', '?')} train / "
                f"{stats.get('rerank_test', '?')} test",
                "labelled `(query, candidate, label)` pairs, featurised",
            ),
            (
                "`golden`",
                str(stats.get("golden", "?")),
                "hand-curated `(query, gold ids)`, never used for training",
            ),
        ],
    )
    return f"""\
---
license: cc-by-4.0
language: [ru, en, tr, zh]
task_categories: [token-classification, text-retrieval]
size_categories: [n<1K]
tags: [geonames, toponym, ner, reranking, synthetic]
configs:
  - config_name: queries
    data_files:
      - split: train
        path: queries/query_dataset.parquet
  - config_name: plan
    data_files:
      - split: train
        path: queries/query_plan.parquet
  - config_name: rerank
    data_files:
      - split: train
        path: rerank/train.parquet
      - split: test
        path: rerank/test.parquet
  - config_name: golden
    data_files:
      - split: test
        path: eval/golden_set.parquet
---

# geosearch-queries

Synthetic multilingual queries naming populated places, with gold GeoNames ids
and character-offset entity spans — plus the training and evaluation sets
derived from them.

{configs}

Also shipped, but deliberately **not** viewer configs:

- `ner/train.json`, `ner/val.json` — the GLiNER fine-tuning export. Its `ner`
  field is `[[token_start, token_end, "LABEL"]]`, a heterogeneous `int, int,
  str` inner list with no honest viewer type.
- `raw/query_dataset.checkpoint.jsonl` — the append-only generation log,
  including `valid=false` rows of a different shape. Kept because it is the only
  record of what the model reliably mangles, and it warm-starts a re-run.

## How the queries were generated

One LLM call per plan row ({stats.get("model", "DeepSeek")}), prompted per query
kind, returning JSON with the query text, its entity spans in the query's own
**inflected surface form**, and a confidence.

The JSON mode enforces no schema server-side, so the contract is enforced
locally: the response is parsed, the span count per type is checked against what
that prompt asked for, and `start`/`end` are filled by string search — never by
the model. A row that fails is retried, then persisted as `valid=false` with a
reason. Only valid rows reach the Parquet.

What the validator deliberately does **not** check: that the city written is the
city requested. Declension changes the surface form (`Москва` → `Москве`), so
demanding the requested spelling would reject correct Russian and Turkish
inflections. The span count is the proxy.

## Gold entity columns

`gold_city_entities` / `gold_country_entities` / `gold_admin1_entities` are the
spans bucketed by the same function the live engine applies to GLiNER output, so
they are directly comparable to what the engine produces. `REGION` and `STATE`
both map to admin1.

These are for **evaluating** NER and filtering dataset quality. The reranker
does not train on them by default — it mines spans from the live pipeline, which
is what keeps train/serve parity.

## Caveats

- **Synthetic.** These are LLM-written queries, not observed user traffic. They
  are shaped to cover query kinds (one city, several cities, city + region, city
  + country) and a stratified population range, which real traffic is not.
- **Small.** ~{stats.get("queries", "?")} queries. Per-language and per-label
  slices rest on tens of spans; read them with their support counts.
- **Disambiguation rows are deliberately hard.** Where a query names a region or
  country, the gold id is narrowed to it, so the name's other homonyms become
  hard negatives on purpose.

{GEONAMES_ATTRIBUTION.format(**_scope(settings))}

The queries themselves are LLM output layered over GeoNames names, released
under the same CC BY 4.0 terms.
"""
