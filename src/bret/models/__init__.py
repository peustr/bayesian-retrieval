from bret.models.bayesian import BayesianBERTRetriever, BayesianDistilBERTRetriever
from bret.models.core import BERTRetriever, DistilBERTRetriever
from bret.utils import get_hf_model_id


def model_factory(model_name, method, device):
    if method == "dpr":
        if model_name.startswith("bert"):
            retriever_class = BERTRetriever
        elif model_name.startswith("distilbert"):
            retriever_class = DistilBERTRetriever
    elif method == "bret":
        if model_name.startswith("bert"):
            retriever_class = BayesianBERTRetriever
        elif model_name.startswith("distilbert"):
            retriever_class = BayesianDistilBERTRetriever
    tokenizer, model = retriever_class.build(get_hf_model_id(model_name), device=device)
    return tokenizer, model
