from torch import nn

_model_registry = {
    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2",
    "bert-mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert-small": "google/bert_uncased_L-4_H-512_A-8",
    "bert-medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert-base": "google/bert_uncased_L-12_H-768_A-12",
    "distilbert-base": "distilbert/distilbert-base-uncased",
    "bert-base-msmarco": "sentence-transformers/msmarco-bert-base-dot-v5",
    "distilbert-base-msmarco-tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
}


def get_hf_model_id(model_name):
    return _model_registry[model_name]


def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transformer_hidden_dim(model: nn.Module) -> int:
    """
    Retrieves the hidden dimension of the last layer of a transformer model.

    Sourced from ChatGPT.

    Args:
        model (nn.Module): The transformer model.

    Returns:
        int: The hidden dimension size of the last layer.
    """
    if hasattr(model, "config"):
        if hasattr(model.config, "hidden_size"):
            return model.config.hidden_size  # For BERT, BERT-Large
        elif hasattr(model.config, "dim"):
            return model.config.dim  # For DistilBERT

    # Fallback: Find the last linear layer
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features

    raise ValueError("Could not determine hidden dimension from the model.")
