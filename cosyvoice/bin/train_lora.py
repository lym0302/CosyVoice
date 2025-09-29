# train_lora.py
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Licensed under the Apache License, Version 2.0

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import math
import torch
import torch.nn as nn
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

from cosyvoice.llm.lora_core import (
    LoRALinear, 
    apply_lora_to_named_linears, 
    only_optimize_lora_parameters, 
    freeze_non_lora, 
    lora_state_dict,
    load_lora_state_dict,
    merge_all_lora)

def get_args():
    parser = argparse.ArgumentParser(description='LoRA training for Qwen2LM (cosyvoice)')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model or base model checkpoint')
    parser.add_argument('--lora_checkpoint', help='lora adapter checkpoint (optional, .pt)')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl', choices=['nccl', 'gloo'], help='distributed backend')
    parser.add_argument('--num_workers', default=0, type=int, help='num workers')
    parser.add_argument('--prefetch', default=100, type=int, help='prefetch')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use AMP')
    parser.add_argument('--timeout', default=60, type=int, help='timeout (s) of cosyvoice_join.')
    # LoRA hyperparams (can also be put in config under train_conf.lora)
    parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank (overrides config)')
    parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA alpha (overrides config)')
    parser.add_argument('--lora_dropout', type=float, default=None, help='LoRA dropout (overrides config)')
    parser.add_argument('--lora_targets', type=str, default=None, help='Comma separated target module names e.g. llm_decoder')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Allow LoRA hyperparams in config under train_conf.lora
    lora_conf = configs['train_conf'].get('lora', {})
    lora_r = args.lora_r if args.lora_r is not None else lora_conf.get('r', 16)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else lora_conf.get('alpha', 32)
    lora_dropout = args.lora_dropout if args.lora_dropout is not None else lora_conf.get('dropout', 0.05)
    lora_targets = args.lora_targets.split(',') if (args.lora_targets is not None) else lora_conf.get('targets', ['llm_decoder'])
    train_lora_lr = lora_conf.get('lr', configs['train_conf'].get('lr', 1e-3))

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(args, configs, gan)

    # Do some sanity checks and save config
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard
    writer = init_summarywriter(args)

    # load base checkpoint (optional)
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if 'step' in state_dict: start_step = state_dict['step']
            if 'epoch' in state_dict: start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # ----------------------------
    # Apply LoRA BEFORE moving to GPU / DDP
    # ----------------------------
    logging.info(f"Apply LoRA: targets={lora_targets}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    model = apply_lora_to_named_linears(model, target_names=lora_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, train_bias=False)

    # If user provided lora checkpoint, load it (must match module names)
    if args.lora_checkpoint is not None and os.path.exists(args.lora_checkpoint):
        lora_state = torch.load(args.lora_checkpoint, map_location='cpu')
        try:
            load_lora_state_dict(model, lora_state, strict=False)
            logging.info(f"Loaded LoRA adapter from {args.lora_checkpoint}")
        except Exception as e:
            logging.warning(f"Failed to load lora checkpoint: {e}")

    # Freeze other params
    freeze_non_lora(model)

    # Dispatch model from cpu to gpu (and DDP/DeepSpeed wrapper)
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler (we call init to keep scheduler logic), then replace optimizer with LoRA-only optimizer
    model_for_init, optimizer_tmp, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    # build optimizer only with LoRA params
    import torch.optim as optim
    optimizer = optim.AdamW(list(only_optimize_lora_parameters(model)), lr=train_lora_lr, weight_decay=0.0)
    # keep scheduler as returned (user may want to step it). If it depends on optimizer, you may need to wrap/adjust.
    # If returned scheduler is None, create a simple one (optional)
    if scheduler is None:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)

    # set scheduler step to start_step if possible
    try:
        scheduler.set_step(start_step)
    except Exception:
        pass
    if scheduler_d is not None:
        try:
            scheduler_d.set_step(start_step)
        except Exception:
            pass

    # Save init checkpoints (base model saved by existing save_model)
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # If rank 0, save initial LoRA adapter
    if dist.get_rank() == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(lora_state_dict(model), os.path.join(args.model_dir, "lora_init.pt"))

    # Executor
    executor = Executor(gan=gan)
    executor.step = start_step

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Training loop (same as original)
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join)
        dist.destroy_process_group(group_join)

        # 每个 epoch 结束 保存 LoRA adapter（只保存可训练的 A/B）
        if dist.get_rank() == 0:
            torch.save(lora_state_dict(model), os.path.join(args.model_dir, f"lora_epoch{epoch}.pt"))
            logging.info(f"Saved lora adapter at epoch {epoch}")

    # 训练结束，保存最终 LoRA
    if dist.get_rank() == 0:
        torch.save(lora_state_dict(model), os.path.join(args.model_dir, "lora_final.pt"))
        logging.info("Saved final lora adapter")

if __name__ == '__main__':
    main()
