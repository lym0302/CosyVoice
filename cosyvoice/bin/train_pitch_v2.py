# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)
from cosyvoice.pitch.pitch_predictor import MultiModalPitchPredictor


device = int(os.environ.get('LOCAL_RANK', 0))

def get_args():
    parser = argparse.ArgumentParser(description='训练MultiModalPitchPredictor网络')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='并行训练引擎')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--train_data', required=True, help='训练数据文件')
    parser.add_argument('--cv_data', required=True, help='交叉验证数据文件')
    parser.add_argument('--checkpoint', help='检查点模型路径')
    parser.add_argument('--model_dir', required=True, help='模型保存目录')
    parser.add_argument('--llm_pretrain_path', required=True, help='llm pretrain path')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard日志目录')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='分布式训练后端')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='数据读取子进程数量')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='预取数量')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='使用固定内存缓冲区')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='使用自动混合精度训练')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='保存模型/优化器状态')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='cosyvoice_join超时时间(秒)')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    gan = False
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    override_dict = {'flow': None, 'hift': None, 'hifigan': None}
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
            
    configs['train_conf'] = configs['train_conf_pitch']
    configs['train_conf'].update(vars(args))

    # 分布式初始化
    init_distributed(args)

    # 数据集和 dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, False)

    # 配置检查和保存
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard
    writer = init_summarywriter(args)

    # 模型加载
    llm_model = configs['llm']
    if args.llm_pretrain_path and os.path.exists(args.llm_pretrain_path):
        state_dict = torch.load(args.llm_pretrain_path, map_location='cpu')
        llm_model.load_state_dict(state_dict, strict=False)
    else:
        logging.warning('llm model {} do not exsist!'.format(args.llm_pretrain_path))

    model = MultiModalPitchPredictor(
        llm=llm_model,
        text_dim=896,
        speech_token_dim=896,
        spk_dim=192,
        hidden_dim=512,
        attention_heads=8,
        dropout=0.1
    )

    start_step, start_epoch = 0, -1
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            start_step = checkpoint.get('step', 0)
            start_epoch = checkpoint.get('epoch', -1)
            logging.info(f'成功加载检查点: {args.checkpoint}')
        else:
            model.load_state_dict(checkpoint, strict=False)
            logging.info(f'成功加载模型状态: {args.checkpoint}')

    # 移动模型到GPU
    model = wrap_cuda_model(args, model)

    # 优化器和调度器
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, gan)
    if scheduler is not None and hasattr(scheduler, "set_step"):
        scheduler.set_step(start_step)

    # 保存初始检查点
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # Get executor
    executor = Executor(gan=gan)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))
    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()
