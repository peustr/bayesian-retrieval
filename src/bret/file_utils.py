import os


def get_checkpoint_file_name(model_dir, model_name, method=None):
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    if method is not None:
        ckpt_file_name = "{}-{}.pt".format(model_name, method)
    else:
        ckpt_file_name = "{}.pt".format(model_name)
    return os.path.join(model_dir, ckpt_file_name)


def get_embedding_file_name(embedding_dir, ckpt_file, data_file):
    if "/" in ckpt_file:
        ckpt_file = ckpt_file.split("/")[-1].split(".")[0]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    embedding_file_name = "{}-{}.pt".format(ckpt_file, data_file)
    return os.path.join(embedding_dir, embedding_file_name)


def get_tokenizer_cache_file_name(tokenizer_cache_dir, tokenizer_name, data_file):
    if "/" in tokenizer_name:
        tokenizer_name = tokenizer_name.split("/")[-1]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    cache_file_name = "{}-{}.cache".format(tokenizer_name, data_file)
    return os.path.join(tokenizer_cache_dir, cache_file_name)
