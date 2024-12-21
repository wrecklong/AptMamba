#!/bin/bash
#SBATCH -o /cluster/home/guosun/shangye/Vim_original/Vim/vim/logs/job.%j.out
#SBATCH -e /cluster/home/guosun/shangye/Vim_original/Vim/vim/logs/job.%j.err
#SBATCH --time 96:00:00
#SBATCH -J vim_t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpumem:32GB
#SBATCH --gpus=8

module load  stack/2024-06  intel-oneapi-compilers/2023.2.0
module load pigz/2.7-q5vkqhu

rsync -aq ./ ${TMPDIR}

cd $TMPDIR

mkdir datasets
tar -I pigz -xf /cluster/work/cvl/guosun/shangye/datasets01/imagenet_full_size.tar.gz -C ${TMPDIR}/datasets

export PYTHONPATH=
source  /cluster/project/cvl/guosun/shangye/Vim/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
                    --use_env main.py   --epochs=300\
                    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2   \
                    --batch-size 128 --input-size 384 --drop-path 0.0  \
                    --weight-decay 0.1  --num_workers 32 --data-path ${TMPDIR}/datasets/imagenet_full_size/  \
                    --output_dir /cluster/work/cvl/guosun/shangye/output/Vim_original/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_300e_batch_size128_input_size384
                   --no_amp
