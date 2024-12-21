import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops

def cal_flops(model, shape=(3, 224, 224), verbose=True):
    # shape = self.__input_shape__[1:]
    supported_ops={
        "aten::silu": None, # as relu is in _IGNORED_OPS
        "aten::neg": None, # as relu is in _IGNORED_OPS
        "aten::exp": None, # as relu is in _IGNORED_OPS
        "aten::flip": None, # as permute is in _IGNORED_OPS
        # "prim::PythonOp.CrossScan": None,
        # "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScanFn": partial(selective_scan_flop_jit),
    }

    model.cuda().eval()

    input = torch.randn((1, *shape), device=next(model.parameters()).device)
    #params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    del model, input
    return sum(Gflops.values()) * 1e9
    return f"params {params} GFLOPs {sum(Gflops.values())}"
 


@torch.no_grad()
def throughput(images, model):
    model.eval()

    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)
