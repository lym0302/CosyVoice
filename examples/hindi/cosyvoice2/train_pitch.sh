#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;


pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B


# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0"
data_dir=datas/v1_1000h-bbc_v1_240
exp_dir=exps/pitch
# model=llm
model=pitch

torchrun --nnodes=1 --nproc_per_node=4 \
	--rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
	cosyvoice/bin/train_pitch_v2.py \
	--train_engine "torch_ddp" \
	--config conf/cosyvoice2_pitch.yaml \
	--train_data $data_dir/train.data.list \
	--cv_data $data_dir/dev.data.list \
	--qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
	--llm_pretrain_path exps/v1_1000h-bbc_v1_240/cosyvoice/llm/torch_ddp/llm_avg5.pt \
	--model_dir $exp_dir/cosyvoice/${model}/torch_ddp \
	--tensorboard_dir $exp_dir/tensorboard/${model}/torch_ddp \
	--ddp.dist_backend nccl \
	--num_workers 2 \
	--prefetch 100 \
	--pin_memory \
	--use_amp \
	--deepspeed_config ./conf/ds_stage2.json \
	--deepspeed.save_states model+optimizer \
	> logs/train_${model}.log 2>&1


# average model
# average_num=3
# decode_checkpoint=$exp_dir/cosyvoice/${model}/torch_ddp/${model}.pt
# echo "do model average and final checkpoint is $decode_checkpoint"
# python cosyvoice/bin/average_model.py \
# 	--dst_model $decode_checkpoint \
# 	--src_path $exp_dir/cosyvoice/${model}/torch_ddp  \
# 	--num ${average_num} \
# 	--val_best



# --checkpoint ${pretrained_model_dir}/${model}.pt \

