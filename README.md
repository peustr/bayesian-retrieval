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

```
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
unzip msmarco.zip
wget https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
```
