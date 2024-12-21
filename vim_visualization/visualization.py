import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models_mamba import VisionMambaPrunning
# build transforms
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
import torch.nn as nn
from utils import batch_index_select

t_resize_crop = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
])

t_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])
# build model
BASE_RATE = 0.7
KEEP_RATE = [BASE_RATE, BASE_RATE ** 2, BASE_RATE ** 3]
CKPT_PATH = '/cluster/work/cvl/guosun/shangye/output/Vim_new/vimpruning_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_clstok_div2_300einit_100e_batch_size128_p0.7_lr0.00001_min_lr1e-6_decoder_pruning_loss3stage_weight0.1_mse_weight0.02_sort_keep_policy_pretrain_mae_inat/checkpoint.pth'


model = VisionMambaPrunning(
    patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=False, num_classes=1010,
    pruning_loc=[6, 12, 18],token_merge_module=None, token_ratio=KEEP_RATE, distill=True, 
    decoder_embed_dim = 512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6))

checkpoint = torch.load(CKPT_PATH, map_location='cpu')['model']
model.load_state_dict(checkpoint)

def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices

def gen_masked_tokens(tokens, indices, alpha=0.2):
    indices = [i for i in range(196) if i not in indices]
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 256
    return tokens

def recover_image(tokens):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(14, 14, 16, 16, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

def gen_visualization(image, decisions):
    keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    image_tokens = image.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, keep_indices[i]))
        for i in range(3)
    ]
    viz = np.concatenate([image] + stages, axis=1)
    return viz

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    if (title != ''):
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    else:
        plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def run_one_image(img, model,device, output_dir):
    x = t_to_tensor(img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    #x = torch.einsum('nhwc->nchw', x)
    x = x.to(device)

    #with torch.cuda.amp.autocast(): 
    output, decisions, pred = model(x)
   
        
    #decisions = [decisions[i].cpu().numpy() for i in range(3)]
     
    tokens_index = torch.arange(196).repeat(1,1).to(device)
    keep_indices = []
    for i in range(3):
        tokens_index = batch_index_select(tokens_index, decisions[i])
        keep_indices.append(tokens_index.squeeze(0))

    img = np.asarray(img)
    image_tokens = img.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)
    
    stage_1 = recover_image(gen_masked_tokens(image_tokens, keep_indices[0]))
    stage_1 = torch.tensor(stage_1)
    stage_2 = recover_image(gen_masked_tokens(image_tokens, keep_indices[1]))
    stage_2 = torch.tensor(stage_2)
    stage_3 = recover_image(gen_masked_tokens(image_tokens, keep_indices[2]))
    stage_3 = torch.tensor(stage_3)

    y = unpatchify(pred,16)  # reconstruct image
    y = torch.einsum('nchw->nhwc', y).detach().cpu() 

    mask = torch.zeros(1, 196, device=device)
    mask = mask.scatter_(1, keep_indices[-1].unsqueeze(0), 1)

    mask = mask.unsqueeze(-1).repeat(1, 1, 16**2*3)
    mask = unpatchify(mask,16)
    mask = torch.einsum('nchw->nhwc', mask).cpu()
    
    x = torch.einsum('nchw->nhwc', x).cpu()

    # MAE reconstruction pasted with visible patches
    im_paste = x * (mask) + y * (1 - mask)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(1, 6, 1)
    #show_image(x[0], "original")
    show_image(img, "")

    plt.subplot(1, 6, 2)
    #show_image(stage_1, "stage-1")
    show_image(stage_1, "")

    plt.subplot(1, 6, 3)
    #show_image(stage_2, "stage-2")
    show_image(stage_2, "")

    plt.subplot(1, 6, 4)
    #show_image(stage_3, "stage-3")
    show_image(stage_3, "")

    plt.subplot(1, 6, 5)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 6, 6)
    show_image(im_paste[0], "reconstruction + visible")
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(output_dir, format='jpg', bbox_inches='tight', pad_inches=0)

    plt.show()


image_paths = glob.glob(f'imgs/*.jpg')


for i in range(len(image_paths)):
    image_path = image_paths[i]
    image = Image.open(image_path)
    #img = image.resize((224, 224))
    #img = np.array(img) / 255.

    #assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    #img = img - imagenet_mean
    #img = img / imagenet_std
    image = t_resize_crop(image)
    #img = t_to_tensor(image)

    device = 'cuda:0'
    model.to(device)
    model.eval()
    
    out_put_path = 'viz_img/' + str(i) + '.jpg'
    run_one_image(image, model, device, out_put_path)

