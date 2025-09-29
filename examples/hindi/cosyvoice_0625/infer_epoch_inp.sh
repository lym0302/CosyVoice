#!/bin/bash
. ./path.sh || exit 1;

stage=8
stop_stage=8

data_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/filelists
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
exp_dir=exp

epoch_inp=$1
gpu_inp=$2


# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# inference
# llm_model=exp/cosyvoice/llm/torch_ddp/epoch_2_whole_infer.pt

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  # for epoch in $(seq 12 1 16); do
  for epoch in $epoch_inp; do
    # python change_pt.py $epoch $exp_dir
    # echo "eeeeeeeeeeeeeeeeeeeepoch : $epoch"
    llm_model=${exp_dir}/cosyvoice/llm/torch_ddp/epoch_${epoch}_whole_infer.pt
    # 判断模型文件是否存在
    if [ ! -f "$llm_model" ]; then
      echo "Error: LLM model file not found: $llm_model"
      exit 1
    fi
    echo "model: $llm_model"
    CUDA_VISIBLE_DEVICES=$gpu_inp python cosyvoice/bin/inference.py --mode sft \
      --gpu 1 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test/parquet/data.list \
      --prompt_utt2data data/test/parquet/utt2data.list \
      --tts_text `pwd`/test280.json \
      --llm_model $llm_model \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir output/output_${exp_dir}/aa \
      --epoch $epoch
  done
fi



