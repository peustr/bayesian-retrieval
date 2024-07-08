import logging
import random

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset

from bret.data_loaders.preprocessors import (
    QueryDocumentCollator,
    TextCollator,
    TextPreProcessor,
    TrainingDataPreProcessor,
)

logger = logging.getLogger(__name__)


def _load_data(data_file):
    if data_file.endswith(".jsonl"):
        data = HuggingFaceDataset.from_json(data_file)
    elif data_file.endswith(".tsv"):
        data = HuggingFaceDataset.from_csv(data_file, delimiter="\t")
    else:
        raise NotImplementedError("Data file with format {} not supported.".format(data_file.split(".")[-1]))
    return data


class TrainingDataset(Dataset):
    def __init__(self, tokenizer, data, max_qry_len=32, max_psg_len=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_qry_len = max_qry_len
        self.max_psg_len = max_psg_len
        self._num_samples = len(self.data)

    def create_one_example(self, text_encoding, is_query=False):
        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=self.max_qry_len if is_query else self.max_psg_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        group = self.data[idx]
        encoded_query = self.create_one_example(group["query"], is_query=True)
        positives = group["pos"]
        negatives = group["neg"]
        pos_psg = positives[random.randint(0, len(positives) - 1)]
        neg_psg = negatives[random.randint(0, len(negatives) - 1)]
        encoded_passages = [self.create_one_example(pos_psg), self.create_one_example(neg_psg)]
        return encoded_query, encoded_passages


class TrainingDataLoader(DataLoader):
    def __init__(self, tokenizer, data_file, max_qry_len=32, max_psg_len=256, batch_size=16, shuffle=True):
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
        )
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=QueryDocumentCollator(tokenizer, max_qry_len=max_qry_len, max_psg_len=max_psg_len),
            drop_last=True,
        )


class TextDataset(Dataset):
    def __init__(self, tokenizer, data, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self._num_samples = len(self.data)

    def create_one_example(self, text_encoding):
        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        group = self.data[idx]
        text_id = group["id"]
        text = group["text"]
        encoded_text = self.create_one_example(text)
        return text_id, encoded_text


class QueryDataLoader(DataLoader):
    def __init__(self, tokenizer, data_file, max_qry_len=32, batch_size=32, shuffle=True):
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
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=TextCollator(tokenizer, max_qry_len),
            drop_last=False,
        )


class CorpusDataLoader(DataLoader):
    def __init__(self, tokenizer, data_file, max_psg_len=256, batch_size=32, shuffle=True):
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
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=TextCollator(tokenizer, max_psg_len),
            drop_last=False,
        )
