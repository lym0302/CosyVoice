stage=1
stop_stage=2

# audio_dir=/data/youtube_dataset/hi-valid-gt-7_rename
# asr_dir=/data/youtube_dataset/hi-valid-gt-7-asr_rename
# audio_dir=/data/youtube_dataset/bbc_0901_0902/valid_rename
# asr_dir=/data/youtube_dataset/bbc_0901_0902/asr_rename
# flist_dir=filelists/bbc_0901_0902
# audio_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/filelists/yoyo_new3spk/audios
# asr_dir=/data/audio_data_and_asr/audio_data_realtime/hi
# flist_dir=filelists/yoyo_new3spk

flist_dir=filelists_v3/yoyo_v2
mkdir -p $flist_dir
flist_hindi=$flist_dir/data_hindi.list

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo ">>>>>>>>>>>>>> generate data.list"
    python data_process/gen_data_new.py \
        --audio_dir $audio_dir \
        --asr_dir $asr_dir \
        --output_file $flist_hindi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo ">>>>>>>>>>>>>> get asr conf."
    python data_process/get_asrconf_thres.py \
        --input_file $flist_hindi \
        --output_file $flist_dir/asrconf_info.txt 
fi

thres=0.7
flist_asrconf=$flist_dir/data_hindi_asrconf_$thres.list

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo ">>>>>>>>>>>>>> Select data with a threshold greater than $thres "
    python data_process/choose_data_asrconf.py \
        --input_file $flist_hindi \
        --output_file $flist_asrconf \
        --threshold $thres
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ">>>>>>>>>>>>>> get spk_info."
    python data_process/get_spk_info.py \
        --input_file $flist_asrconf \
        --output_file $flist_dir/spk_info_asrconf_$thres.txt 
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ">>>>>>>>>>>>>> Split train, dev, test "
    cp $flist_asrconf $flist_dir/data.list
    python data_process/split_dev_test.py \
        --input_file $flist_dir/data.list \
        --train_file $flist_dir/train_pre.list \
        --dev_file $flist_dir/dev_pre.list \
        --test_file $flist_dir/test_pre.list \
        --dev_ratio 0.05 \
        --test_ratio 0.05 \
        --seed 42 \
        --max_limit 2 \
        --count_thres 50
    
    # cp $flist_dir/train_pre.list $flist_dir/train.list
    # cp $flist_dir/dev_pre.list $flist_dir/dev.list
    # cp $flist_dir/test_pre.list $flist_dir/test.list
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo ">>>>>>>>>>>>>> remove sensitive line."
    for x in train dev test; do
        python data_process/remove_sensitive.py \
            --input_file $flist_dir/${x}_pre.list \
            --output_file $flist_dir/${x}_rmsenti.list
        # cp $flist_dir/${x}_rmsenti.list $flist_dir/${x}.list
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo ">>>>>>>>>>>>>> remove num line."
    for x in train dev test; do
        python data_process/remove_num.py \
            --input_file $flist_dir/${x}_rmsenti.list \
            --output_file $flist_dir/${x}_rmnum.list
        # cp $flist_dir/${x}_rmnum.list $flist_dir/${x}.list
    done
fi


