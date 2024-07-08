from transformers import DataCollatorWithPadding


class TextPreProcessor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x):
        text = self.tokenizer.encode(x["text"], add_special_tokens=False, max_length=self.max_len, truncation=True)
        return {"id": x["id"], "text": text}


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


class TextCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        max_len,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)
        self.max_len = max_len

    def __call__(self, batch):
        ids = [item[0] for item in batch]
        txts = [item[1] for item in batch]
        if isinstance(ids[0], list):
            ids = sum(ids, [])
        if isinstance(txts[0], list):
            txts = sum(txts, [])
        txts_collated = self.tokenizer.pad(
            txts,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return ids, txts_collated


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

    def __call__(self, batch):
        qrys = [item[0] for item in batch]
        psgs = [item[1] for item in batch]
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
