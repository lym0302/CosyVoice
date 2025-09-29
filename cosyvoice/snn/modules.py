# coding=utf-8

import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 1.0  # Default firing threshold

# Custom activation function for LIF neurons
class ActFun(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, thresh=1.0, alpha=0.5):
        # Step function for spike generation
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.alpha = alpha
        return input.ge(thresh).float()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Rectangular surrogate gradient for backward pass
        (input,) = ctx.saved_tensors
        thresh = ctx.thresh
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < alpha
        temp = temp / (2 * alpha)
        return grad_input * temp.float(), None, None

def act_fun(input, thresh=1.0, alpha=0.5):
    return ActFun.apply(input, thresh, alpha)

class mem_update(nn.Module):
    """LIF (Leaky Integrate-and-Fire) Layer implementation"""
    def __init__(self, decay=0.25, thresh=1.0, alpha=0.5):
        super(mem_update, self).__init__()
        self.decay = decay
        self.thresh = thresh
        self.alpha = alpha

    def forward(self, x):
        time_window = x.size()[0]  # Time steps
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        
        for i in range(time_window):
            if i >= 1:
                # Update membrane potential
                mem = mem_old * self.decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem, self.thresh, self.alpha)
            mem_old = mem.clone()
            output[i] = spike
        return output

# Surrogate gradient functions
def g_window(x,alpha):
    """Rectangular surrogate gradient"""
    temp = abs(x) < alpha
    return temp / (2 * alpha)

def g_sigmoid(x,alpha):
    """Sigmoid surrogate gradient"""
    sgax = (alpha*x).sigmoid()
    return alpha * (1-sgax) * sgax

def g_atan(x,alpha):
    """Arctangent surrogate gradient"""
    return alpha / (2 * (1 + ((np.pi / 2) * alpha * x)**2))

def g_gaussian(x,alpha):
    """Gaussian surrogate gradient"""
    return (1 / np.sqrt(2 * np.pi * alpha**2)) * torch.exp(-x**2 / (2 * alpha**2))

# Multi-synaptic activation functions
class ActFun_rectangular(torch.autograd.Function):
    """Multi-synaptic activation function with rectangular surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=0.5):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        # Create multiple firing thresholds
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_window(input-thresholds[0],alpha)+g_window(input-(thresholds[1]),alpha)+g_window(input-(thresholds[2]),alpha)+g_window(input-(thresholds[3]),alpha))
 
        return grad_x, None, None, None

def act_fun_rectangular(input, init_thre=1.0, D=4, alpha=0.5):
    return ActFun_rectangular.apply(input, init_thre, D, alpha)

class ActFun_sigmoid(torch.autograd.Function):
    """Multi-synaptic activation function with sigmoid surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=4.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_sigmoid(input-thresholds[0],alpha)+g_sigmoid(input-thresholds[1],alpha)+g_sigmoid(input-thresholds[2],alpha)+g_sigmoid(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_sigmoid(input, init_thre=1.0, D=4, alpha=4.0):
    return ActFun_sigmoid.apply(input, init_thre, D, alpha)

class ActFun_atan(torch.autograd.Function):
    """Multi-synaptic activation function with arctangent surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_atan(input-thresholds[0],alpha)+g_atan(input-thresholds[1],alpha)+g_atan(input-thresholds[2],alpha)+g_atan(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_atan(input, init_thre=1.0, D=4, alpha=2.0):
    return ActFun_atan.apply(input, init_thre, D, alpha)

class ActFun_gaussian(torch.autograd.Function):
    """Multi-synaptic activation function with Gaussian surrogate gradient"""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, init_thre=1.0, D=4, alpha=0.4):
        ctx.save_for_backward(input)
        ctx.init_thre = init_thre
        ctx.D = D
        ctx.alpha = alpha
        
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        out = input.ge(thresholds[0]).float() + input.ge(thresholds[1]).float() + input.ge(thresholds[2]).float() + input.ge(thresholds[3]).float()
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        init_thre = ctx.init_thre
        D = ctx.D
        alpha = ctx.alpha
        
        grad_input = grad_output.clone()
        thresholds = torch.arange(D, device=input.device).float() + init_thre
        grad_x = grad_input * (g_gaussian(input-thresholds[0],alpha)+g_gaussian(input-thresholds[1],alpha)+g_gaussian(input-thresholds[2],alpha)+g_gaussian(input-thresholds[3],alpha))
 
        return grad_x, None, None, None    
    
def act_fun_gaussian(input, init_thre=1.0, D=4, alpha=0.4):
    return ActFun_gaussian.apply(input, init_thre, D, alpha)

class mem_update_MSF(nn.Module):
    """MSF Layer implementation"""
    def __init__(self, decay=0.25, init_thre=1.0, D=4, surro_gate='rectangular'):
        super(mem_update_MSF, self).__init__()
        self.decay = decay
        self.init_thre = init_thre
        self.D = D
        self.surro_gate = surro_gate
        
        # Dictionary of activation functions
        self.act_fun_dict = {
            'rectangular': act_fun_rectangular,
            'sigmoid': act_fun_sigmoid,
            'atan': act_fun_atan,
            'gaussian': act_fun_gaussian
        }

    def forward(self, x):
        time_window = x.size()[0] ### set timewindow
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        
        # Select activation function
        act_fun = self.act_fun_dict.get(self.surro_gate, act_fun_rectangular)
        
        for i in range(time_window):
            if i >= 1:
                mask = spike > 0
                mem = mem_old * self.decay * (1 - mask.float()) + x[i]
            else:
                mem = x[i]
            # multi-threshold firing function
            spike = act_fun(mem, self.init_thre, self.D)
            mem_old = mem.clone()
            output[i] = spike
        return output