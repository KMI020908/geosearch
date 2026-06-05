"""Fine-tune the retriever model and build a FAISS index of city descriptions.

Usage
-----
python -m src.models.train_retriever

# Skip training, only rebuild FAISS index from an already-trained model:
python -m src.models.train_retriever --index-only

# Custom paths / hyperparams:
python -m src.models.train_retriever --epochs 2 --batch-size 128 --lr 1e-5

Outputs
-------
models/retriever/processed/          — fine-tuned sentence-transformers model
models/retriever/processed/faiss.index  — FAISS IndexFlatIP (cosine via L2-norm)
models/retriever/processed/index_meta.pkl — {idx_to_gid: list[int], cities: dict}
"""
import logging
import polars as pl
from datasets import Dataset
from src.data.languages import get_language
from src.data.utils import make_region_resolver
from country_list import countries_for_language
from src.data.config import PROCESSED_DIR, PRIMARY_LANGUAGES, BIENCODER_MODEL_RAW, BIENCODER_MODEL, MODEL_DIR
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import mine_hard_negatives

logger = logging.getLogger(__name__)


def make_training_args(output_dir: str, epochs: int):
    """
    Returns training args.
    Batch sampler switches based on whether we have explicit negatives or not.
    """
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        metric_for_best_model="eval_ir-eval_cosine_recall@50",
        greater_is_better=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES
    )


def run_round(
    model: SentenceTransformer,
    dataset: Dataset,
    evaluator: InformationRetrievalEvaluator,
    num: int = 1,
    eval_dataset: Dataset | None = None
) -> tuple:

    loss = MultipleNegativesRankingLoss(model)
    args = make_training_args(
        output_dir=str(MODEL_DIR / "biencoder" / "checkpoints" / f"round{num}"),
        epochs=1,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    results = evaluator(model)
    return trainer, results


def main():
    df = pl.read_parquet(PROCESSED_DIR / "cities.parquet")
    region_resolver = make_region_resolver()

    city_descriptions = []
    for row in df.iter_rows(named=True):
        iso_code = PRIMARY_LANGUAGES.get(row["country_code"], "en")
        country_code = row["country_code"]
        country_name = dict(countries_for_language(iso_code))[country_code]
        admin1_name = region_resolver(country_code, row["admin1_code"], iso_code)
        for i in row["info"]:
            if i["language"] == iso_code:
                info = i
                break
            elif i["language"] == "en":
                en_info = i
        else:
            info = en_info
        name = info["names"][0]
        other_names = info["names"][1:]
        city_descriptions.append((
            get_language(iso_code).city_description(name, country_name, admin1_name, other_names),
            row["geoname_id"]
        ))
    city_descriptions = (
        pl.DataFrame(city_descriptions, schema={"description": pl.String(), "geoname_id": pl.Int64()}, orient="row")
        .with_columns(pl.col("description").str.to_lowercase())
    )

    corpus = dict(zip(city_descriptions["geoname_id"].cast(pl.String()).to_list(), city_descriptions["description"].to_list()))

    train_df = (
        pl.read_parquet(PROCESSED_DIR / "retriever_datasets" / "train.parquet")
        .with_columns(pl.col("query").str.to_lowercase())
        .explode("geoname_ids")
        .rename({"geoname_ids": "geoname_id"})
        .join(city_descriptions, on="geoname_id")
    )

    val_data = (
        pl.read_parquet(PROCESSED_DIR / "retriever_datasets" / "val.parquet")
        .with_columns(pl.col("query").str.to_lowercase())
    )

    val_df = (
        val_data
        .explode("geoname_ids")
        .rename({"geoname_ids": "geoname_id"})
        .join(city_descriptions, on="geoname_id")
    )

    dataset_train = Dataset.from_dict({"anchor": train_df["query"].to_list(), "positive": train_df["description"].to_list()})
    dataset_val = Dataset.from_dict({"anchor": val_df["query"].to_list(), "positive": val_df["description"].to_list()})

    logger.info(f"Train size: {train_df.shape[0]} | Eval size: {val_data.shape[0]}")

    queries_val = {}
    relevant_docs_val = {}
    for i, row in enumerate(val_data.iter_rows(named=True)):
        queries_val[str(i)] = row["query"]
        relevant_docs_val[str(i)] = set([str(gid) for gid in row["geoname_ids"]])

    evaluator = InformationRetrievalEvaluator(
        queries=queries_val,
        corpus=corpus,
        relevant_docs=relevant_docs_val,
        precision_recall_at_k=[1, 5, 25, 50],
        accuracy_at_k=[1],
        ndcg_at_k=[10],
        mrr_at_k=[10, 25],
        map_at_k=[10],
        name="ir-eval",
        show_progress_bar=True,
        batch_size=32,
        corpus_chunk_size=50_000
    )

    model = SentenceTransformer(str(BIENCODER_MODEL_RAW))

    run_round(model, dataset_train, evaluator, 1, dataset_val)

    train_with_negatives = mine_hard_negatives(
        dataset=dataset_train,
        model=model,
        corpus=list(corpus.values()),  # The pool of documents to mine negatives from
        num_negatives=3,  # How many hard negatives to attach per (anchor, positive) pair
        relative_margin=0.1,  # Prevents near-positives from being used as negatives
        range_max=50,  # Search within this many top candidates
        sampling_strategy="top",  # "top" = hardest negatives | "random" = more diverse
        batch_size=64,  # Batch size for encoding during mining
    )

    trainer_round2, _ = run_round(model, dataset_train, evaluator, 2, dataset_val)
    trainer_round2.save_model(str(BIENCODER_MODEL))

    logger.info("All done.")


if __name__ == "__main__":
    main()
