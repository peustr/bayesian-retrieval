import logging
import random

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import RandomSampler
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)


class TrainingDataPreProcessor:
    def __init__(self, tokenizer, max_qry_len=32, max_psg_len=256):
        self.tokenizer = tokenizer
        self.max_qry_len = max_qry_len
        self.max_psg_len = max_psg_len

    def __call__(self, x):
        query = self.tokenizer.encode(
            x["query"], add_special_tokens=False, max_length=self.max_qry_len, truncation=True
        )
        positives = []
        for pos in x["pos"]:
            positives.append(
                self.tokenizer.encode(pos, add_special_tokens=False, max_length=self.max_psg_len, truncation=True)
            )
        negatives = []
        for neg in x["neg"]:
            negatives.append(
                self.tokenizer.encode(neg, add_special_tokens=False, max_length=self.max_psg_len, truncation=True)
            )
        return {"query": query, "pos": positives, "neg": negatives}


class TrainingDataset(PyTorchDataset):
    def __init__(self, tokenizer, data, max_qry_len=32, max_psg_len=256, num_train_psg=8):
        self.tokenizer = tokenizer
        self.data = data
        self.max_qry_len = max_qry_len
        self.max_psg_len = max_psg_len
        self.num_train_psg = num_train_psg
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
        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)
        encoded_passages = []
        positives = group["pos"]
        negatives = group["neg"]
        pos_psg = positives[random.randint(0, len(positives) - 1)]
        encoded_passages.append(self.create_one_example(pos_psg))
        num_negatives = self.num_train_psg - 1
        if len(negatives) < num_negatives:
            neg_psgs = random.choices(negatives, k=num_negatives)
        else:
            random.shuffle(negatives)
            neg_psgs = negatives[:num_negatives]
        for neg_psg in neg_psgs:
            encoded_passages.append(self.create_one_example(neg_psg))
        return encoded_query, encoded_passages


class QueryDocumentCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
        max_qry_len=32,
        max_psg_len=256,
    ):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)
        self.max_qry_len = max_qry_len
        self.max_psg_len = max_psg_len

    def __call__(self, features):
        qrys = [f[0] for f in features]
        psgs = [f[1] for f in features]
        if isinstance(qrys[0], list):
            qrys = sum(qrys, [])
        if isinstance(psgs[0], list):
            psgs = sum(psgs, [])
        qry_collated = self.tokenizer.pad(
            qrys,
            padding="max_length",
            max_length=self.max_qry_len,
            return_tensors="pt",
        )
        psg_collated = self.tokenizer.pad(
            psgs,
            padding="max_length",
            max_length=self.max_psg_len,
            return_tensors="pt",
        )
        return qry_collated, psg_collated


def make_data_loader(
    tokenizer,
    data_file,
    max_qry_len=32,
    max_psg_len=256,
    num_train_qry=8,
    num_train_psg=8,
    split="train",
    tokenizer_cache_file_name=None,
):
    if split == "train":
        preprocessor_cls = TrainingDataPreProcessor
        dataset_cls = TrainingDataset
    else:
        raise NotImplementedError("Val/Test data loaders not yet supported.")
    data = HFDataset.from_json(data_file)
    tokenized_data = data.map(
        preprocessor_cls(tokenizer, max_qry_len, max_psg_len),
        batched=False,
        remove_columns=data.column_names,
        load_from_cache_file=True,
        cache_file_name=tokenizer_cache_file_name,
        desc="Running tokenizer on the training data.",
    )
    dataset = dataset_cls(
        tokenizer,
        tokenized_data,
        max_qry_len=max_qry_len,
        max_psg_len=max_psg_len,
        num_train_psg=num_train_psg,
    )
    return DataLoader(
        dataset,
        batch_size=num_train_qry,
        sampler=RandomSampler(dataset),
        collate_fn=QueryDocumentCollator(tokenizer, max_qry_len=max_qry_len, max_psg_len=max_psg_len),
        drop_last=True,
    )
