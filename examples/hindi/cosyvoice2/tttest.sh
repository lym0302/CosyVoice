CUDA_VISIBLE_DEVICES=0 nohup python test_infer_batch_lora.py -m trained_models/test_1min_basethres300avg5_lora -o output_test1min/test_1min_basethres300avg5_lora > aaaa.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python test_infer_batch_lora.py -m trained_models/test_1min_basethres300avg5_sft -o output_test1min/test_1min_basethres300avg5_sft > bbbb.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test_infer_batch_lora.py -m trained_models/test_1min_baseyoyo6spkavg5_lora -o output_test1min/test_1min_baseyoyo6spkavg5_lora > cccc.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python test_infer_batch_lora.py -m trained_models/test_1min_baseyoyo6spkavg5_sft -o output_test1min/test_1min_baseyoyo6spkavg5_sft > dddd.log 2>&1 &
