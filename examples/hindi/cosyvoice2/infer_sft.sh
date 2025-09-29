#!/bin/bash
. ./path.sh || exit 1;

stage=8
stop_stage=8


pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B
exp_name=yoyo_sft
exp_dir=exps/${exp_name}
data_dir=datas/${exp_name}
model=llm


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  # for epoch in $(seq 0 1 11); do
  for epoch in 0 1 5 10; do
    python change_pt.py $epoch $exp_dir $model
    echo "eeeeeeeeeeeeeeeeeeeepoch : $epoch"
    model_path=${exp_dir}/cosyvoice/${model}/torch_ddp/epoch_${epoch}_whole_infer.pt
    # 判断模型文件是否存在
  # for num in 5; do
    # model_path=${exp_dir}/cosyvoice/${model}/torch_ddp/llm_avg${num}.pt
    if [ ! -f "$model_path" ]; then
      echo "Error: LLM model file not found: $model_path"
      exit 1
    fi
    echo "model: $model_path"
    CUDA_VISIBLE_DEVICES=0 python cosyvoice/bin/inference.py --mode zz \
      --gpu 0 \
      --config conf/cosyvoice2.yaml \
      --prompt_data $data_dir/test/parquet/data.list \
      --prompt_utt2data $data_dir/test/parquet/utt2data.list \
      --tts_text $data_dir/test.json \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --llm_model $model_path \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir output/output_${exp_name}/aa \
      --epoch ${epoch}
  done
fi


# 测试一下 speech token 在预训练模型下的效果对不对
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for mode in tttest; do
  echo "mmmmmmmmmmmmmmmmmmmmode: $mode"
  # for mode in zero_shot; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 1 \
      --config conf/cosyvoice2.yaml \
      --prompt_data $data_dir/test/parquet/data.list \
      --prompt_utt2data $data_dir/test/parquet/utt2data.list \
      --tts_text $data_dir/test_part.json \
      --llm_model $pretrained_model_dir/llm.pt \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir output/output_pretrained/aa \
      --epoch pretrained
  done
fi


