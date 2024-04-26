import logging

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from bret.data_loaders.preprocessors import (
    QueryDocumentCollator,
    TextCollator,
    TextDataset,
    TextPreProcessor,
    TrainingDataPreProcessor,
    TrainingDataset,
)

logger = logging.getLogger(__name__)


def _load_data(data_file):
    if data_file.endswith(".jsonl"):
        data = HFDataset.from_json(data_file)
    elif data_file.endswith(".tsv"):
        data = HFDataset.from_csv(data_file, delimiter="\t")
    else:
        raise NotImplementedError("Data file with format {} not supported.".format(data_file.split(".")[-1]))
    return data


def make_training_data_loader(
    tokenizer,
    data_file,
    max_qry_len=32,
    max_psg_len=256,
    num_train_qry=8,
    num_train_psg=8,
    shuffle=True,
):
    data = _load_data(data_file)
    tokenized_data = data.map(
        TrainingDataPreProcessor(tokenizer, max_qry_len, max_psg_len),
        batched=False,
        remove_columns=data.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on the training data.",
    )
    dataset = TrainingDataset(
        tokenizer,
        tokenized_data,
        max_qry_len=max_qry_len,
        max_psg_len=max_psg_len,
        num_train_psg=num_train_psg,
    )
    return DataLoader(
        dataset,
        batch_size=num_train_qry,
        shuffle=shuffle,
        collate_fn=QueryDocumentCollator(tokenizer, max_qry_len=max_qry_len, max_psg_len=max_psg_len),
        drop_last=True,
    )


def make_query_data_loader(
    tokenizer,
    data_file,
    max_qry_len=32,
    batch_size=32,
    shuffle=True,
):
    data = _load_data(data_file)
    tokenized_data = data.map(
        TextPreProcessor(tokenizer, max_qry_len),
        batched=False,
        remove_columns=data.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on the queries.",
    )
    dataset = TextDataset(
        tokenizer,
        tokenized_data,
        max_len=max_qry_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TextCollator(tokenizer, max_qry_len),
        drop_last=False,
    )


def make_corpus_data_loader(
    tokenizer,
    data_file,
    max_psg_len=256,
    batch_size=32,
    shuffle=True,
):
    data = _load_data(data_file)
    tokenized_data = data.map(
        TextPreProcessor(tokenizer, max_psg_len),
        batched=False,
        remove_columns=data.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on the corpus.",
    )
    dataset = TextDataset(
        tokenizer,
        tokenized_data,
        max_len=max_psg_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TextCollator(tokenizer, max_psg_len),
        drop_last=False,
    )
