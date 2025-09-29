stage=0
stop_stage=5

audio_dir=/data/youtube_dataset/hi-valid-gt-7_rename
asr_dir=/data/youtube_dataset/hi-valid-gt-7-asr_rename
flist_dir=filelists_temp/bbc_v1_240
mkdir -p $flist_dir
flist_hindi=$flist_dir/data_hindi.list

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo ">>>>>>>>>>>>>> generate data.list"
    python data_process/gen_data.py \
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
        --output_file $flist_dir/spk_info.txt 
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ">>>>>>>>>>>>>> Split train, dev, test "
    cp $flist_asrconf $flist_dir/data.list
    python data_process/split_dev_test.py \
        --input_file $flist_dir/data.list \
        --train_file $flist_dir/train.list \
        --dev_file $flist_dir/dev.list \
        --test_file $flist_dir/test.list \
        --dev_ratio 0.05 \
        --test_ratio 0.05 \
        --seed 42 \
        --max_limit 2 \
        --count_thres 50
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo ">>>>>>>>>>>>>> remove num line."
    for x in train dev test; do
        python data_process/remove_num.py \
            --input_file $flist_dir/$x.list \
            --output_file $flist_dir/${x}_rmnum.list
    done
fi
