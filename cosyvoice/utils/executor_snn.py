# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    update_parameter_and_lr, log_per_step, log_per_save, 
    batch_forward, batch_backward, save_model, cosyvoice_join
)
from cosyvoice.utils.mask import make_pad_mask


class ExecutorSNN(Executor):
    """SNN增强的训练执行器"""

    def __init__(self, snn_config=None):
        super().__init__(gan=False)  # SNN训练不使用GAN
        self.snn_config = snn_config or {}
        self.training_mode = self.snn_config.get('training_mode', 'snn_only')
        self.strength_schedule = self.snn_config.get('strength_schedule', False)
        self.log_freq = self.snn_config.get('log_freq', 100)
        
        logging.info(f"🧠 SNN Executor initialized with mode: {self.training_mode}")

    def update_snn_training_mode(self, model, epoch, total_epochs):
        """根据epoch更新SNN训练模式"""
        if self.training_mode == 'gradual':
            # 渐进式训练：前50%只训练SNN，后50%端到端微调
            if hasattr(model, 'progressive_training_schedule'):
                model.progressive_training_schedule(epoch, total_epochs)
            elif hasattr(model.module, 'progressive_training_schedule'):  # DDP wrapped
                model.module.progressive_training_schedule(epoch, total_epochs)
        elif self.training_mode == 'end_to_end':
            # 端到端训练：解冻所有权重进行训练
            if hasattr(model, 'set_end_to_end_training'):
                model.set_end_to_end_training()
            elif hasattr(model.module, 'set_end_to_end_training'):  # DDP wrapped
                model.module.set_end_to_end_training()
        elif self.training_mode == 'snn_only':
            # 只训练SNN：冻结预训练权重
            if hasattr(model, 'set_snn_only_training'):
                model.set_snn_only_training()
            elif hasattr(model.module, 'set_snn_only_training'):  # DDP wrapped
                model.module.set_snn_only_training()

    def log_snn_statistics(self, model, writer, step):
        """记录SNN相关统计信息"""
        if step % self.log_freq == 0:
            # 获取SNN强度信息
            if hasattr(model, 'get_training_status'):
                status = model.get_training_status()
            elif hasattr(model.module, 'get_training_status'):  # DDP wrapped
                status = model.module.get_training_status()
            else:
                return

            # 记录到tensorboard
            if writer is not None:
                if 'snn_strengths' in status:
                    for layer_idx, strength in status['snn_strengths'].items():
                        writer.add_scalar(f'SNN/strength_layer_{layer_idx}', strength, step)
                
                if 'trainable_ratio' in status:
                    writer.add_scalar('SNN/trainable_params_ratio', status['trainable_ratio'], step)

            # 打印统计信息
            if self.rank == 0:
                logging.info(f"🔍 Step {step} SNN Status: "
                           f"Mode={status.get('training_mode', 'unknown')}, "
                           f"Trainable={status.get('trainable_params', 0):,}/"
                           f"{status.get('total_params', 0):,} "
                           f"({status.get('trainable_ratio', 0):.1%})")

    def batch_forward_snn(self, model, batch_dict, scaler, info_dict):
        """SNN特有的前向传播"""
        # 先执行标准前向传播
        info_dict = batch_forward(model, batch_dict, scaler, info_dict)
        
        # 如果模型有SNN状态重置方法，在每个batch开始时重置
        if hasattr(model, 'reset_snn_states'):
            model.reset_snn_states()
        elif hasattr(model.module, 'reset_snn_states'):
            model.module.reset_snn_states()
        
        return info_dict

    def train_one_epoch_snn(self, model, optimizer, scheduler, train_data_loader, 
                           cv_data_loader, writer, info_dict, scaler, group_join):
        """SNN专用的一个epoch训练"""
        
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'🧠 Epoch {self.epoch} SNN TRAIN info lr {lr} rank {self.rank}')
        logging.info(f'using accumulate grad, new batch size is {info_dict["accum_grad"]} times larger than before')
        
        # 更新SNN训练模式
        total_epochs = info_dict.get('max_epoch', 100)
        self.update_snn_training_mode(model, self.epoch, total_epochs)
        
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                
                if cosyvoice_join(group_join, info_dict):
                    break

                # Gradient synchronization handling
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                else:
                    context = nullcontext

                with context():
                    # SNN特有的前向传播
                    info_dict = self.batch_forward_snn(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                
                # 记录SNN统计信息
                self.log_snn_statistics(model, writer, self.step)
                
                # Save per step if needed
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv_snn(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
                    
        dist.barrier()
        save_model_path = self.cv_snn(model, cv_data_loader, writer, info_dict, on_batch_end=True)
        return save_model_path

    @torch.inference_mode()
    def cv_snn(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        """SNN模型的交叉验证"""
        model.eval()
        
        logging.info(f'🔍 Epoch {self.epoch} CV info lr {info_dict.get("lr", 0)} rank {self.rank}')
        
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        
        # 在验证时重置SNN状态
        if hasattr(model, 'reset_snn_states'):
            model.reset_snn_states()
        elif hasattr(model.module, 'reset_snn_states'):
            model.module.reset_snn_states()

        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            
            # 前向传播
            batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_dict.items()}
            
            try:
                outputs = model(batch_dict, self.device)
                loss = outputs['loss']
                acc = outputs.get('acc', 0.0)
                
                total_loss += loss.item()
                total_acc += acc.item() if torch.is_tensor(acc) else acc
                total_batches += 1
                
            except Exception as e:
                logging.warning(f"CV batch {batch_idx} failed: {e}")
                continue
            
            # 限制验证batch数量
            if batch_idx >= 50:  # 只验证前50个batch
                break

        # 计算平均值
        if total_batches > 0:
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches
            
            # 记录到tensorboard
            if writer is not None:
                writer.add_scalar('CV/loss', avg_loss, self.step)
                writer.add_scalar('CV/acc', avg_acc, self.step)
            
            # 记录SNN统计信息到验证
            self.log_snn_statistics(model, writer, self.step)
            
            if self.rank == 0:
                logging.info(f'📊 Epoch {self.epoch} CV Results: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')

        # 保存模型
        save_model_path = None
        if on_batch_end and self.rank == 0:
            save_dict = {
                'epoch': self.epoch,
                'step': self.step,
                'lr': info_dict.get('lr', 0),
                'cv_loss': avg_loss if total_batches > 0 else float('inf'),
                'cv_acc': avg_acc if total_batches > 0 else 0.0,
                'model_dir': info_dict['model_dir'],
                'train_engine': info_dict.get('train_engine', 'torch_ddp')
            }
            save_model_path = save_model(model, f'epoch_{self.epoch}', save_dict)
            logging.info(f'💾 Model saved to: {save_model_path}')

        return save_model_path

    def get_snn_model_info(self, model):
        """获取SNN模型信息"""
        if hasattr(model, 'get_training_status'):
            return model.get_training_status()
        elif hasattr(model.module, 'get_training_status'):
            return model.module.get_training_status()
        else:
            return {'training_mode': 'unknown', 'total_params': 0, 'trainable_params': 0}