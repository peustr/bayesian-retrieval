import os


def get_checkpoint_file_name(model_dir, model_name, method=None):
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


def get_results_file_name(results_dir, ckpt_file, data_file, k):
    if "/" in ckpt_file:
        ckpt_file = ckpt_file.split("/")[-1].split(".")[0]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    results_file_name = "run-{}-{}-k={}.pt".format(ckpt_file, data_file, k)
    return os.path.join(results_dir, results_file_name)
