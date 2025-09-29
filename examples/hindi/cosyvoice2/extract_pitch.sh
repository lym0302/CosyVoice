data_name=bbc_0723_0811
nohup python tools/extract_pitch_v2.py -d datas/${data_name}/train/ -o datas/${data_name}/f0 -n 10 > logs/extract_pitch_${data_name}_train.log 2>&1 &
nohup python tools/extract_pitch_v2.py -d datas/${data_name}/dev/ -o datas/${data_name}/f0 -n 10 > logs/extract_pitch_${data_name}_dev.log 2>&1 &
nohup python tools/extract_pitch_v2.py -d datas/${data_name}/test/ -o datas/${data_name}/f0 -n 10 > logs/extract_pitch_${data_name}_test.log 2>&1 &
