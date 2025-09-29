#!/bin/bash
# 这个脚本是为了测试指标的 wer, ss, dnsmos 全流程

source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
stage=0
stop_stage=5

exp_dir=exp_base11
gen_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_$exp_dir
test_json=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/test80.json
raw_audio_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_infer

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "=== Stage 0: Azure ASR ==="
    conda activate audio_recorder
    cd /data/liangyunming/others/asr
    echo -e "y\ny" | python batch_asr.py --audio_dir $gen_dir --lang hi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "=== Stage 1: Generate ASR data.list ==="
    conda activate cosyvoice
    cd /data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625
    python data_process/gen_data.py \
        --audio_dir $gen_dir \
        --asr_dir ${gen_dir}_asr \
        --output_file ${gen_dir}.list
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "=== Stage 2: Get WER ==="
    python get_wer.py \
        --real_file $test_json \
        --pred_file ${gen_dir}.list \
        --output_file ${gen_dir}_wer.csv
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "=== Stage 3: Get Speaker Similarity (SS) ==="
    conda activate 3D-Speaker
    gen_audio_file=${gen_dir}_wav.scp
    ls ${gen_dir}/aa/*.wav > $gen_audio_file
    python get_ss.py \
        --real_file $test_json \
        --raw_audio_dir ${raw_audio_dir}/aa \
        --gen_audio_dir ${gen_dir}/aa \
        --gen_audio_file $gen_audio_file \
        --output_file ${gen_dir}_ss.csv
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "=== Stage 4: Get per-wav DNSMOS ==="
    conda activate dnsmos36
    cd /data/liangyunming/others/DNS-Challenge/DNSMOS
    python dnsmos_local.py \
        --testset_dir ${gen_dir}/aa \
        --csv_path ${gen_dir}_dnsmos.csv
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "=== Stage 5: Get DNSMOS average ==="
    conda activate cosyvoice
    cd /data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625
    python get_dnsmos.py \
        --json_file $test_json \
        --csv_file ${gen_dir}_dnsmos.csv \
        --output_file ${gen_dir}_dnsmos_avg.csv
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "=== Stage 6: Copy gen audio ==="
    mv $gen_dir/aa/*.wav ${raw_audio_dir}/aa/
    
fi
