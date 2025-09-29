#!/bin/bash
. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="0,1,2,3"
data_dir=data_yoyo_sft
exp_dir=exp_yoyo_sft_basebbc240_epoch17

torchrun --nnodes=1 --nproc_per_node=4 \
	--rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
	cosyvoice/bin/train.py \
	--train_engine "torch_ddp" \
	--config conf/cosyvoice_sft.yaml \
	--train_data $data_dir/train.data.list \
	--cv_data $data_dir/dev.data.list \
	--model "llm" \
	--checkpoint $exp_dir/cosyvoice/llm/torch_ddp/epoch_17_whole.pt \
	--model_dir $exp_dir/cosyvoice/llm/torch_ddp \
	--tensorboard_dir $exp_dir/tensorboard/llm/torch_ddp \
	--ddp.dist_backend nccl \
	--num_workers 2 \
	--prefetch 100 \
	--pin_memory --use_amp \
	--deepspeed_config ./conf/ds_stage2.json \
	--deepspeed.save_states model+optimizer \
	> train.log 2>&1
