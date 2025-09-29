base_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2

for x in train dev test; do
    python data_process/gen_text_add_emo.py \
        --emo_csv_file ${base_dir}/filelists/bbc07230902_yoyo0904_thres300/emo.csv \
        --wav_scp_file ${base_dir}/datas/bbc07230902_yoyo0904_thres300_emo/${x}/wav.scp \
        --raw_text_file ${base_dir}/datas/bbc07230902_yoyo0904_thres300/${x}/text \
        --new_text_file ${base_dir}/datas/bbc07230902_yoyo0904_thres300_emo/${x}/text
done
