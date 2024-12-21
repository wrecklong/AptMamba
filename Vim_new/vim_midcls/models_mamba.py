# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

import torchvision.models
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
from timm.models.vision_transformer import Block as Block_attn
import math

from collections import namedtuple
from utils import batch_index_select
from utils  import get_2d_sincos_pos_embed

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
def sort_by_policy_mixer(hidden_states, policy, mixer, inference_params):
    B, N, C = hidden_states.shape
    cls_token = hidden_states[:,0:1,:]
    masked_hidden_states_withoutcls = hidden_states[:,1:,] * policy #policy without cls

    token_position = policy.squeeze(-1).sum(dim=-1).long()
    token_position = token_position.min() // 2 # position for cls token

    sorted_indices = torch.argsort(policy.squeeze(-1), dim=1, descending=True) 
    sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, C)
    #set reserved token at the front of the sequence
    sorted_masked_hidden_states_withoutcls = torch.gather(masked_hidden_states_withoutcls, dim=1, index=sorted_indices_expanded)

    # place cls token
    sorted_masked_hidden_states = torch.cat([sorted_masked_hidden_states_withoutcls[:,0:token_position,], cls_token, sorted_masked_hidden_states_withoutcls[:, token_position:, ]],dim=1)
    
    hidden_states = mixer(sorted_masked_hidden_states, inference_params=inference_params)
    

    inverse_indices = torch.argsort(sorted_indices, dim=1)
    inverse_indices_expanded = inverse_indices.unsqueeze(-1).expand(-1, -1, C)
    
    hidden_states_withoutcls = torch.cat([hidden_states[:, :token_position,], hidden_states[:, token_position + 1:,]], dim = 1)
    cls_token = hidden_states[:,token_position].unsqueeze(1)
    
    # resotre the position of tokens (without cls token)
    hidden_states_withoutcls = torch.gather(hidden_states_withoutcls, dim=1, index=inverse_indices_expanded)

    hidden_states = torch.cat([cls_token,hidden_states_withoutcls], dim=1)
    
    return hidden_states


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, pos_embed=None, policy=None, score_predictor = None, p_count=None,
        prev_decision = None, early_hidden_states = None, token_merge_module = None, num_keep_node=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                ) 


        if score_predictor is not None:
            pruning_loss = 0
            if self.training:
                B = hidden_states.shape[0]
        
                fused_hidden_states = hidden_states + early_hidden_states
                spatial_hidden_states = fused_hidden_states[:,1:,]
                cls_token = fused_hidden_states[:,:1, ]
                pred_score = score_predictor[p_count](spatial_hidden_states, cls_token,prev_decision)
                pred_score = pred_score.reshape(B,-1, 2)
                hard_keep_decision = torch.nn.functional.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                #cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                #policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                
                current_pruned_decision = (1-hard_keep_decision) * prev_decision
                if (p_count < 3):
                    pruning_loss = early_pruning_loss(hidden_states, hard_keep_decision, current_pruned_decision)
                
                #else:
                #    pruning_loss = late_pruning_loss(hidden_states, hard_keep_decision, current_pruned_decision)
                
                if token_merge_module is not None:
                    spatial_hidden_states = token_merge_module(
                        spatial_hidden_states, None, hard_keep_decision, current_pruned_decision, None)
                    hidden_states = torch.cat([hidden_states[:, :1, :], spatial_hidden_states], dim=1)
 
                hidden_states = sort_by_policy_mixer(hidden_states, hard_keep_decision, self.mixer, inference_params)

                return hidden_states, residual, hard_keep_decision, pruning_loss

            else:
                B,N,_ = hidden_states.shape
                token_position = N // 2

                fused_hidden_states = hidden_states + early_hidden_states 

                spatial_hidden_states = torch.cat([fused_hidden_states[:,0:token_position,:], fused_hidden_states[:,token_position+1:,:]],dim=1)      
                
                cls_token = fused_hidden_states[:,token_position,:].unsqueeze(1)        

                pred_score = score_predictor[p_count](spatial_hidden_states, cls_token,prev_decision)                           
                pred_score = pred_score.reshape(B,-1, 2)
         
                spatial_hidden_states = torch.cat([hidden_states[:,0:token_position,:], hidden_states[:,token_position+1:,:]], dim=1)


                score = pred_score[:,:,0]
 
                keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                keep_policy, _ = torch.sort(keep_policy)

                spatial_residual = torch.cat([residual[:,0:token_position,:], residual[:,token_position+1:,:]], dim=1)
                spatial_hidden_states = batch_index_select(spatial_hidden_states, keep_policy) #prune without cls
                spatial_residual = batch_index_select(spatial_residual, keep_policy)
                
                cls_token_h = hidden_states[:, token_position,:].unsqueeze(1) #get original cls token
                cls_token_r = residual[:, token_position,:].unsqueeze(1) 

                prev_decision = batch_index_select(prev_decision, keep_policy)
                
                token_position = (num_keep_node + 1) // 2
                hidden_states = torch.cat([spatial_hidden_states[:,0:token_position], cls_token_h, spatial_hidden_states[:,token_position:]],dim=1)
                residual = torch.cat([spatial_residual[:,0:token_position],cls_token_r, spatial_residual[:,token_position:]],dim=1)

                hidden_states = self.mixer(hidden_states, inference_params=inference_params)
         
         
                return hidden_states, residual, prev_decision, #pos_embed
            
        else:
            if policy is not None:
                hidden_states = sort_by_policy_mixer(hidden_states, policy, self.mixer, inference_params)

            else:
                hidden_states = self.mixer(hidden_states, inference_params=inference_params)
              

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class PredictorLG(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv_local = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU()
        )

        self.in_conv_cls = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, cls_t, policy):
        local_x = self.in_conv_local(x)
        cls_t = self.in_conv_cls(cls_t)
        B, N, C = x.size()
        #local_x = x[:,:, :C//2]
        #global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, cls_t.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

def get_sim(x, y, eps=1e-6, mask_eye=-100, l2_norm=True):

    if y is None:
        y = x
    if l2_norm:
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        y = y / (y.norm(dim=-1, keepdim=True) + eps)

    sim = torch.bmm(x, y.permute(0, 2, 1))
    if mask_eye is not None:
        sim.masked_fill_(
            torch.eye(x.size(1), device=x.device).unsqueeze(0).bool(), mask_eye)
    return sim


class TPSModule(nn.Module):

    # from pruned tokens to keep tokens
    def __init__(self, l2_norm=True, temperature=1) -> None:
        super().__init__()
        self.l2_norm = l2_norm
        self.temperature = temperature

    def forward(self, x, y, current_keep_decision, current_pruned_decision, relative_dist=None):
        B, N, C = x.size(0), x.size(1), x.size(2)
        if self.training:

            cos_sim = get_sim(
                x, None, mask_eye=-100, l2_norm=self.l2_norm)

            cos_sim = cos_sim/self.temperature
            cos_sim = cos_sim.masked_fill(
                ~current_keep_decision.bool().reshape(B, 1, N), -100)

            sim_th = cos_sim.amax(
                dim=2, keepdims=True)

            # N, pruned token dim, keep token dim
            mask = (cos_sim == sim_th).float() * current_pruned_decision
            cos_sim = (mask * cos_sim)
            # N,keep token dim, pruned_token dim
            mask = mask.permute(0, 2, 1)
            cos_sim = cos_sim.permute(0, 2, 1)
            numerator = torch.exp(cos_sim) * mask
            denominator = math.e + numerator.sum(dim=-1, keepdims=True)
            x = x * (math.e / denominator) + \
                torch.bmm(numerator / denominator, x)

        else:

            # given k =  prune num
            cos_sim = get_sim(
                y, x, mask_eye=None, l2_norm=self.l2_norm)
            cos_sim = cos_sim/self.temperature
            sim_th = cos_sim.amax(dim=2, keepdims=True)
            mask = (cos_sim == sim_th).float()
            # N, pruned token dim, keep token dim
            cos_sim = mask * cos_sim
            # N,keep token dim, pruned_token dim
            mask = mask.permute(0, 2, 1)
            cos_sim = cos_sim.permute(0, 2, 1)
            numerator = torch.exp(cos_sim) * mask
            denominator = math.e + numerator.sum(dim=-1, keepdims=True)
            x = x * (math.e / denominator) + \
                torch.bmm(numerator / denominator, y)

        return x


class MLP_MIXER(nn.Module):
    def __init__(self, dim, num_tokens, expand_factor, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.inner_dim = num_tokens * expand_factor
        self.layer1 = nn.Conv1d(in_channels=num_tokens, out_channels=num_tokens * expand_factor,kernel_size = 1)
        self.layer2 = nn.Conv1d(in_channels=num_tokens * expand_factor, out_channels=num_tokens,kernel_size = 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()  

    
    def forward(self, x, residual):
        x = self.norm(x) + residual
        residual = x
        #x = torch.permute(x, (0, 2, 1))
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        #x = torch.permute(x, (0, 2, 1))

        return x, residual        


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        x = self.pool(x) - x
        x = torch.permute(x, (0,2,1))
        return x


class MAE_Decoder(nn.Module):
    def __init__(self,
                 embed_dim = 192,
                 num_patches = 196,
                 patch_size = 16,
                 decoder_embed_dim = 512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 in_chans=3
                 ):
        # MAE decoder specifics
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block_attn(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
    
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
    
    def forward(self, x, hard_keep_decision):
        # add mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1]-1, 1)
        x_ = x[:, 1:,:] * hard_keep_decision + mask_tokens * (1 - hard_keep_decision)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        mask = (1 - mask.squeeze(-1))
        target = self.patchify(imgs)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    
        return loss

def early_pruning_loss(hidden_states, current_keep_decision, current_pruned_decision,temperature=1.0):
    hidden_states = hidden_states[:,1:,]
    B, N, C = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
    cos_sim = get_sim(
                hidden_states, None, mask_eye=0, l2_norm=True)

    cos_sim = cos_sim/temperature
    # cos_sim = cos_sim.masked_fill(
    #             ~current_keep_decision.bool().reshape(B, 1, N), 0)
    
    cos_sim = cos_sim * current_pruned_decision
    cos_sim = cos_sim * current_keep_decision.reshape(B, 1, N)

    loss = (cos_sim**2).sum(dim=-1) / current_keep_decision.reshape(B,1,N).sum(dim=-1)
    loss = loss.sum(dim = -1, keepdim=True) / current_pruned_decision.reshape(B,1,N).sum(dim = -1)
    
    loss = loss.mean()

    return loss

def late_pruning_loss(hidden_states, current_keep_decision, current_pruned_decision, temperature=1.0):
    hidden_states = hidden_states[:,1:,]
    B, N, C = hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
    cos_sim = get_sim(
                hidden_states, None, mask_eye=0, l2_norm=True)

    cos_sim = cos_sim/temperature
    # cos_sim = cos_sim.masked_fill(
    #             ~current_keep_decision.bool().reshape(B, 1, N), 0)
    
    cos_sim = (1.0-cos_sim) * current_pruned_decision
    cos_sim = cos_sim * current_keep_decision.reshape(B, 1, N)

    loss = (cos_sim).sum(dim=-1) / current_keep_decision.reshape(B,1,N).sum(dim=-1)
    loss = loss.sum(dim = -1, keepdim=True) / current_pruned_decision.reshape(B,1,N).sum(dim = -1)
    loss = loss.mean()

    return loss

class VisionMambaPrunning(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_divide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 pruning_loc = None,
                 token_merge_module=None,
                 token_ratio = None,
                 distill=False,
                 decoder_embed_dim = 512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # add
        self.distill = distill
        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio
        self.token_merge_module = token_merge_module if token_merge_module is not None else None
        
        # predict score of token
        predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]

        self.score_predictor = nn.ModuleList(predictor_list)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # self.decoder = MAE_Decoder(embed_dim=embed_dim,num_patches=num_patches,
        #                            patch_size=patch_size,decoder_embed_dim=decoder_embed_dim,
        #                            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        #                            mlp_ratio=mlp_ratio,norm_layer=norm_layer,in_chans=3)
        # mlp_mixer_list = [MLP_MIXER(embed_dim, num_patches + self.num_tokens, 2) for _ in range(pruning_loc[0])]

        # self.mlp_mixer = nn.ModuleList(mlp_mixer_list)
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # self.decoder.apply(self.decoder._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True
        
        init_n = 14 * 14
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + 1, 1, dtype=x.dtype, device=x.device)
        p_count = 0
        out_pred_prob = []
        last_pruned_features = None

        # mamba impl
        residual = None
        hidden_states = x
        pos_embedding = self.pos_embed
        _, N, C = self.pos_embed.size()
        pos_embedding = pos_embedding.expand(B, N, C)
        pruning_loss = 0
        if not self.if_bidirectional:
            if not self.training:
                hidden_states = torch.cat([hidden_states[:, 1:init_n // 2 + 1], hidden_states[:, 0].unsqueeze(1), hidden_states[:, init_n // 2 + 1:]], dim=1)
                token_position = init_n // 2

            for i, layer in enumerate(self.layers):
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if (i in [1,7,13]):
                    early_hidden_states = hidden_states
                # add pruning token
                if i in self.pruning_loc:
                    #if self.training:
                    #   spatial_hidden_states = hidden_states[:, 1:]
                    #else:
                    # spatial_hidden_states = torch.cat([hidden_states[:, 0:token_position], hidden_states[:, token_position+1:]],dim=1)
                    # spatial_hidden_states = hidden_states[:, 1:]
                    # pred_score = self.score_predictor[p_count](spatial_hidden_states, prev_decision).reshape(B, -1, 2)
                    
                    if self.training:
                        hidden_states, residual, hard_keep_decision, stage_pruning_loss = layer(
                            hidden_states, residual, inference_params=inference_params, pos_embed = None, policy=None, score_predictor = self.score_predictor, p_count=p_count,
                            prev_decision = prev_decision, early_hidden_states = early_hidden_states,token_merge_module=None, 
                        )
                        
                        pruning_loss += stage_pruning_loss

                        last_pruned_features = hidden_states
                        out_pred_prob.append(hard_keep_decision.reshape(B, init_n))

                        cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                        policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
       
                        prev_decision = hard_keep_decision

       
                    else:
                        num_keep_node = int(init_n * self.token_ratio[p_count])

                        hidden_states, residual, prev_decision = layer(
                            hidden_states, residual, inference_params=inference_params, pos_embed=None, policy=None, score_predictor = self.score_predictor, p_count=p_count,
                            prev_decision = prev_decision, early_hidden_states = early_hidden_states, token_merge_module=None, num_keep_node=num_keep_node
                        )
        
                    p_count += 1

                else:
                    if self.training:
                        hidden_states, residual = layer(
                            hidden_states, residual, inference_params=inference_params, policy=prev_decision
                        )
                    else:
                        hidden_states, residual = layer(
                            hidden_states, residual, inference_params=inference_params
                        )
                    
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        # if self.if_cls_token:
        #     if self.use_double_cls_token:
        #         return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
        #     else:
        #         if self.use_middle_cls_token:
        #             return hidden_states[:, token_position, :]
        #         elif if_random_cls_token_position:
        #             return hidden_states[:, token_position, :]
        #         else:
        #             return hidden_states[:, token_position, :]

        # if self.final_pool_type == 'none':
        #     return hidden_states[:, -1, :]
        # elif self.final_pool_type == 'mean':
        #     return hidden_states.mean(dim=1)
        # elif self.final_pool_type == 'max':
        #     return hidden_states
        # elif self.final_pool_type == 'all':
        #     return hidden_states
        # else:
        #     raise NotImplementedError
        if not self.training:
           B,N,_ = hidden_states.shape
           token_position = N // 2
           cls_token = hidden_states[:, token_position,:].unsqueeze(1)
           hidden_states = torch.cat([cls_token, hidden_states[:,0:token_position,:], hidden_states[:,token_position + 1:,:]],dim=1)


        return hidden_states,prev_decision,out_pred_prob, last_pruned_features, pruning_loss

    def forward(self, x, return_features=True, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        imgs = x
        x, prev_decision, out_pred_prob,last_pruned_features, pruning_loss = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        
        # if self.training:
        #     pred = self.decoder(last_pruned_features, prev_decision)
        #     mse_loss = self.decoder.forward_loss(imgs, pred, prev_decision)
        
        features = x[:, 1:]
        x = x[:, 0]
        x = self.head(x)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob, pruning_loss
            else:
                return x, out_pred_prob, pruning_loss
        # if self.final_pool_type == 'max':
        #     x = x.max(dim=1)[0]
        else:
            return x


class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_divide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)


        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        features  =  torch.cat([hidden_states[:, :token_position], hidden_states[:, token_position + 1:]], dim = 1)
        return hidden_states[:, token_position, :],  features
        # return only cls token if it exists
        # if self.if_cls_token:
        #     if self.use_double_cls_token:
        #         return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
        #     else:
        #         if self.use_middle_cls_token:
        #             return hidden_states[:, token_position, :]
        #         elif if_random_cls_token_position:
        #             return hidden_states[:, token_position, :]
        #         else:
        #             return hidden_states[:, token_position, :]

        # if self.final_pool_type == 'none':
        #     return hidden_states[:, -1, :]
        # elif self.final_pool_type == 'mean':
        #     return hidden_states.mean(dim=1)
        # elif self.final_pool_type == 'max':
        #     return hidden_states
        # elif self.final_pool_type == 'all':
        #     return hidden_states
        # else:
        #     raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        cls_t, token_t = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        
        cls_t = self.head(cls_t)
        # if self.final_pool_type == 'max':
        #     x = x.max(dim=1)[0]
        return cls_t,token_t


@register_model
def vimpruning_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    keep_rate = kwargs['keep_rate'] 
    model = VisionMambaPrunning(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=False, 
        pruning_loc=[6, 12, 18],token_merge_module=TPSModule(temperature=1), token_ratio=keep_rate, distill=True, 
        decoder_embed_dim = 512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vimpruning_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_clstok_div2(pretrained=False, **kwargs):
    keep_rate = kwargs['keep_rate'] 
    model = VisionMambaPrunning(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=False,
        pruning_loc=[6, 12, 18], token_merge_module=TPSModule(temperature=1),token_ratio=keep_rate, distill=True, 
        decoder_embed_dim = 512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet_101(pretrained=False, **kwargs):
    model = torchvision.models.resnet101(pretrained)
    return model

