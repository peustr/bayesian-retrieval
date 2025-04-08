


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=10:00:00 python src/bret/scripts/train_mcd.py \
    --dataset_id msmarco \
    --training_data_file data/msmarco-train.jsonl \
    --model_name distilbert-base \
    --num_samples 3 \
    --batch_size 64 \
    --num_epochs .001 \
    --lr 0.000005 \
    --min_lr 0.00000005 \
    --warmup_rate 0.1 \
    --max_qry_len 32 \
    --max_psg_len 256 \
    --output_dir output/trained_encoders



srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=10:00:00  