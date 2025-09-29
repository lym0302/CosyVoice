#!/bin/bash
. ./path.sh || exit 1;

stage=8
stop_stage=8

data_dir=data_yoyo_sft
# data_dir=data
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
exp_dir=exp_yoyo_sft_basebbc240_epoch17
model=llm


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for epoch in $(seq 17 1 19); do
  # for epoch in 29; do
    python change_pt.py $epoch $exp_dir $model
    echo "eeeeeeeeeeeeeeeeeeeepoch : $epoch"
    model_path=${exp_dir}/cosyvoice/${model}/torch_ddp/epoch_${epoch}_whole_infer.pt
    # 判断模型文件是否存在
    if [ ! -f "$model_path" ]; then
      echo "Error: LLM model file not found: $model_path"
      exit 1
    fi
    echo "model: $model_path"
    CUDA_VISIBLE_DEVICES=0 python cosyvoice/bin/inference.py --mode sft \
      --gpu 1 \
      --config conf/cosyvoice.yaml \
      --prompt_data $data_dir/test/parquet/data.list \
      --prompt_utt2data $data_dir/test/parquet/utt2data.list \
      --tts_text $data_dir/test.json \
      --llm_model $model_path \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir output/output_${exp_dir}/aa \
      --epoch $epoch
  done
fi



