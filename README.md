# Bayesian Retrieval (BRET)

## Installation

Create a `conda` environment.
```
conda create -n bret python=3.12
```

Install the requirements.
```
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
```

FAISS is also a requirement, but it only supports `conda` installation. :sleeping:
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

(Optional) Install this repository in editable mode, to make development easier.
```
pip install -e .
```

(Optional) Install `black` and `isort` for automatic code formatting.
```
pip install black isort
```
You can set up commit hooks, but if you find commit hooks annoying, you may run them manually:
```
black -l 120 src
isort --profile black src
```

## Download MS-MARCO

Create a `./data` folder under the root folder and download MS-MARCO. We also use the hard negatives from the SentenceTransformers repository.
```
mkdir data
cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
unzip msmarco.zip
wget https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
```

## How to train a DPR model

After having downloaded MS-MARCO along with the hard negatives, run:
```
python scripts/prepare_msmarco_training_data.py
```
which will generate `./data/msmarco-train.jsonl`. One should also prepare the corpus file. Run:
```
python src/bret/scripts/prepare_corpus_file.py --dataset_id msmarco --split dev
```
to generate `./data/msmarco-corpus.jsonl`.
Then, running a command like
```
python src/bret/scripts/train_dpr.py \
    --dataset_id msmarco \
    --training_data_file data/msmarco-train.jsonl \
    --model_name bert-base \
    --num_samples 1 \
    --batch_size 16 \
    --num_epochs 4 \
    --lr 0.000005 \
    --min_lr 0.00000005 \
    --warmup_rate 0.1 \
    --max_qry_len 32 \
    --max_psg_len 256 \
    --output_dir output/trained_encoders
```
will train a basic DPR model; in this example, it uses a `bert-base` backbone.

**Note:** This script includes validation logic. To create a validation set, I randomly sample 10,000 documents from the corpus and store them in `./data/msmarco-corpus-val.jsonl`, and 100 queries from the training set and store them in `./data/msmarco-val.jsonl`, making sure that the relevant document for each of these queries is in the validation corpus. Finally, I remove these queries from the training set to avoid data leakage.

## Bayesian models

The Bayesian models in this repository are (experimental) work in progress, and not yet published. Please, experiment with caution.
