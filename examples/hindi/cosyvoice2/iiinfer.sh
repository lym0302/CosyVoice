base_dir=/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2
pretrained_model=/data/liangyunming/tts_20250618/CosyVoice/pretrained_models/CosyVoice2-0.5B
# data_name=bbc07230902_yoyo0904_thres480
data_name=$1
trained_model=${base_dir}/trained_models/${data_name}
save_model_dir=${base_dir}/exp_sft/${data_name}/cosyvoice/llm/torch_ddp


stage=4
stop_stage=4

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Create a trained model dir..."
    rm -r ${trained_model}
    mkdir -p ${trained_model}
    ln -s ${pretrained_model}/* ${trained_model}/
    echo "Enroll spk"
    python enroll_spk_auto.py ${data_name}
    ln -s ${base_dir}/datas/${data_name}/enroll/spk2info.pt ${trained_model}/
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Get avg llm model"
    for average_num in 5 3 2; do
        python cosyvoice/bin/average_model.py \
            --dst_model ${save_model_dir}/llm_avg${average_num}.pt \
            --src_path ${save_model_dir}  \
            --num ${average_num} \
            --val_best
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Infer avg model"
    for epoch in 5 3 2; do
    # for epoch in 3; do
        rm ${trained_model}/llm.pt
        ln -s ${save_model_dir}/llm_avg${epoch}.pt ${trained_model}/llm.pt
        python test_infer_batch.py -m ${trained_model} -o output3/${data_name}/${data_name}_avg${epoch}/aa

    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Infer single model"
    for epoch in 1 3 2 4; do
    # for epoch in 2; do
        # python change_pt.py ${save_model_dir}/epoch_${epoch}_whole.pt ${save_model_dir}/epoch_${epoch}_whole_infer.pt
        rm ${trained_model}/llm.pt
        ln -s ${save_model_dir}/epoch_${epoch}_whole_infer.pt ${trained_model}/llm.pt
        python test_infer_batch.py -m ${trained_model} -o output3/${data_name}/${data_name}_epoch${epoch}/aa
    done 
fi


