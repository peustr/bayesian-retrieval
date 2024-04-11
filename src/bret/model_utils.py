_model_registry = {
    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2",
    "bert-mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert-small": "google/bert_uncased_L-4_H-512_A-8",
    "bert-medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert-base": "google/bert_uncased_L-12_H-768_A-12",
}


def get_hf_model_id(model_name):
    return _model_registry[model_name]