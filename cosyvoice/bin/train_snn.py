# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 SNN Enhancement (authors: Claude, User)
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

from cosyvoice.utils.executor_snn import ExecutorSNN
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your SNN-enhanced network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    
    # SNN specific arguments
    parser.add_argument('--snn_training_mode',
                        default='snn_only',
                        choices=['snn_only', 'end_to_end', 'gradual'],
                        help='SNN training mode')
    parser.add_argument('--snn_layer_indices',
                        type=str,
                        default=None,
                        help='Comma-separated list of layer indices to apply SNN (e.g., "10,11,12,13")')
    parser.add_argument('--snn_strength_schedule',
                        action='store_true',
                        default=False,
                        help='Use progressive SNN strength scheduling')
    parser.add_argument('--snn_log_freq',
                        type=int,
                        default=100,
                        help='Frequency to log SNN statistics')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    # SNN training is specifically for LLM models
    if args.model != 'llm':
        raise ValueError("SNN training currently only supports 'llm' model type")
    
    
    
    # Process SNN layer indices
    snn_layer_indices = None
    if args.snn_layer_indices:
        snn_layer_indices = [int(x.strip()) for x in args.snn_layer_indices.split(',')]
        logging.info(f"🎯 SNN will be applied to layers: {snn_layer_indices}")

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    print("11111111111111111111111 ", args.config)
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    
    # Add SNN configurations
    if 'snn_config' not in configs:
        configs['snn_config'] = {}
        configs['snn_config'].update({
            'training_mode': args.snn_training_mode,
            'layer_indices': snn_layer_indices,
            'strength_schedule': args.snn_strength_schedule,
            'log_freq': args.snn_log_freq
        })
    
    configs['train_conf'].update(vars(args))
    
    logging.info(f"🧠 Starting SNN-enhanced training with mode: {configs['snn_config']['training_mode']}")

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader (no GAN for SNN)
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, False)

    # Do some sanity checks and save config to args.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Load model
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    
    # Load checkpoint if provided - SNN specific handling
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            
            # For SNN models, we may need special handling
            if hasattr(model, 'load_pretrained_weights'):
                # Use custom loading method if available
                model.load_pretrained_weights(state_dict)
                logging.info("🔄 Loaded pretrained weights with SNN-specific handling")
            else:
                # Standard loading with non-strict mode for SNN adapters
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logging.info(f"⚠️  Missing keys (expected for new SNN modules): {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"❌ Unexpected keys: {unexpected_keys}")
            
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} does not exist!'.format(args.checkpoint))

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler (no GAN optimizers for SNN)
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, False)
    scheduler.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # Get SNN-specific executor
    executor = ExecutorSNN(
        snn_config=configs['snn_config']
    )
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    logging.info(f'🚀 Starting SNN training from step {start_step}, epoch {start_epoch}')
    
    # Log SNN model status
    if hasattr(model, 'get_training_status'):
        status = model.get_training_status()
        logging.info(f"📊 SNN Training Status: {status}")
    elif hasattr(model.module, 'get_training_status'):  # DDP wrapped
        status = model.module.get_training_status()
    
    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        
        # SNN-specific training loop
        executor.train_one_epoch_snn(
            model, optimizer, scheduler, 
            train_data_loader, cv_data_loader,
            writer, info_dict, scaler, group_join
        )
        
        dist.destroy_process_group(group_join)


if __name__ == '__main__':
    logging.info("🧠 SNN-Enhanced CosyVoice Training Started")
    main()
