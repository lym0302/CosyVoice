stage=3
stop_stage=5

flist_dir=filelists
data_file=$flist_dir/data.list

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ">>>>>>>>>>>>>> get spk_info."
    python data_process/get_spk_info.py \
        --input_file $data_file \
        --output_file $flist_dir/spk_info.txt 
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ">>>>>>>>>>>>>> Split train, dev, test "
    python data_process/split_dev_test.py \
        --input_file $flist_dir/data.list \
        --train_file $flist_dir/train.list \
        --dev_file $flist_dir/dev.list \
        --test_file $flist_dir/test.list \
        --dev_ratio 0.05 \
        --test_ratio 0.05 \
        --seed 42 \
        --max_limit 200
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo ">>>>>>>>>>>>>> remove num line."
    for x in train dev test; do
        python data_process/remove_num.py \
            --input_file $flist_dir/$x.list \
            --output_file $flist_dir/${x}_rmnum.list
    done
fi
