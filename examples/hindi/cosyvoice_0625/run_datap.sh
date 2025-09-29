#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# 新添数据的处理
. ./path.sh || exit 1;

stage=3
stop_stage=4

data_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/filelists_temp/bbc_240
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
save_dir=data_bbc_240

# use cpu
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train dev test; do
    mkdir -p $save_dir/$x
    python local/prepare_data.py --src_dir $data_dir/$x.list --des_dir $save_dir/$x
  done
fi


# use cpu
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in train dev test; do
    echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in $save_dir/$x dir"
    tools/extract_embedding.py --dir $save_dir/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi


# use gpu
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for x in train dev test; do
    echo "Extract discrete speech token, you will get utt2speech_token.pt in $save_dir/$x dir"
    tools/extract_speech_token.py --dir $save_dir/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train dev test; do
    mkdir -p $save_dir/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $save_dir/$x \
      --des_dir $save_dir/$x/parquet
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare train.data.list"
  for x in train dev; do
    ls $save_dir/$x/parquet/*.tar > $save_dir/$x.data.list
  done
fi


