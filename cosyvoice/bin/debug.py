# coding=utf-8
import json
from cosyvoice.utils.file_utils import read_lists, read_json_lists
from hyperpyyaml import load_hyperpyyaml
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--config', default="conf/cosyvoice.yaml", help='config file')
    parser.add_argument('--prompt_data', default="data/test/parquet/data.list", help='prompt data file')
    parser.add_argument('--prompt_utt2data', default="data/test/parquet/utt2data.list", help='prompt data file')
    parser.add_argument('--tts_text', default="test280.json", help='tts input file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--llm_model', default="../../../pretrained_models/CosyVoice-300M/llm.pt", help='llm model file')
    parser.add_argument('--flow_model', default="../../../pretrained_models/CosyVoice-300M/flow.pt", help='flow model file')
    parser.add_argument('--hifigan_model', default="../../../pretrained_models/CosyVoice-300M/hift.pt", help='hifigan model file')
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot', 'tttest'],
                        help='inference mode')
    parser.add_argument('--result_dir', default="asd", help='asr result file')
    parser.add_argument('--epoch', default="aaa", help='epoch')
    args = parser.parse_args()
    print(args)
    return args


def Dataset(data_list_file,
            data_pipeline,
            mode='train',
            gan=False,
            shuffle=True,
            partition=True,
            tts_file='',
            prompt_utt2data=''):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert mode in ['train', 'inference']
    lists = read_lists(data_list_file)
    print("11111111111111111: ", len(lists), lists[0])
    if mode == 'inference':
        with open(tts_file) as f:
            tts_data = json.load(f)
        utt2lists = read_json_lists(prompt_utt2data)
        print("2222222222222222222: ", len(utt2lists))
        # filter unnecessary file in inference mode
        lists = list({utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists})
        print("333333333333333333: ", len(lists), lists[0])
        

args = get_args()
with open(args.config, 'r') as f:
    configs = load_hyperpyyaml(f)
test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False,
                           tts_file=args.tts_text, prompt_utt2data=args.prompt_utt2data)