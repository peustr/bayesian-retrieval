import os


def get_checkpoint_file_name(model_dir, model_name, method=None):
    if method is not None:
        ckpt_file_name = "{}-{}.pt".format(model_name, method)
    else:
        ckpt_file_name = "{}.pt".format(model_name)
    return os.path.join(model_dir, ckpt_file_name)


def get_embedding_file_name(embedding_dir, ckpt_file, data_file, shard_index=None):
    if "/" in ckpt_file:
        ckpt_file = ckpt_file.split("/")[-1].split(".")[0]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    if shard_index is None:
        embedding_file_name = "{}-{}.pt".format(ckpt_file, data_file)
    else:
        embedding_file_name = "{}-{}-s{:02d}.pt".format(ckpt_file, data_file, shard_index)
    return os.path.join(embedding_dir, embedding_file_name)


def get_run_file_name(run_dir, ckpt_file, data_file, k):
    if "/" in ckpt_file:
        ckpt_file = ckpt_file.split("/")[-1].split(".")[0]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    run_file_name = "{}-{}-k={:02d}.json".format(ckpt_file, data_file, k)
    return os.path.join(run_dir, run_file_name)


def get_results_file_name(results_dir, ckpt_file, data_file, k):
    if "/" in ckpt_file:
        ckpt_file = ckpt_file.split("/")[-1].split(".")[0]
    if "/" in data_file:
        data_file = data_file.split("/")[-1].split(".")[0]
    results_file_name = "run-{}-{}-k={:02d}.json".format(ckpt_file, data_file, k)
    return os.path.join(results_dir, results_file_name)
