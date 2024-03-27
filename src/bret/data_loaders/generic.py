import csv
import json
import logging
import os

from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class GenericDataLoader:
    def __init__(
        self, data_dir, corpus_file="corpus.jsonl", query_file="queries.jsonl", qrels_dir="qrels", split="train"
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.corpus_file = os.path.join(data_dir, corpus_file)
        self.query_file = os.path.join(data_dir, query_file)
        self.qrels_dir = os.path.join(data_dir, qrels_dir)
        self.qrels_file = os.path.join(self.qrels_dir, split + ".tsv")
        self.split = split

    @staticmethod
    def check(f_in, ext):
        if not os.path.exists(f_in):
            raise ValueError(f"File {f_in} not present! Please provide accurate file.")
        if not f_in.endswith(ext):
            raise ValueError(f"File {f_in} must be present with extension {ext}.")

    def load(self):
        self.check(self.corpus_file, "jsonl")
        self.check(self.query_file, "jsonl")
        self.check(self.qrels_file, "tsv")
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), self.split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), self.split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        return self.corpus, self.queries, self.qrels

    def load_corpus(self):
        self.check(self.corpus_file, "jsonl")
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
