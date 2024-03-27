import os


def get_checkpoint_file_name(model_dir, model_name, method=None):
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    if method is not None:
        model_name = model_name + "-" + method
    return os.path.join(model_dir, model_name + ".pt")


def get_tokenizer_cache_file_name(tokenizer_cache_dir, tokenizer_name):
    if "/" in tokenizer_name:
        tokenizer_name = tokenizer_name.split("/")[-1]
    return os.path.join(tokenizer_cache_dir, tokenizer_name + ".cache")
