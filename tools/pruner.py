import torch
import os.path as osp
import argparse
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-I', type=str, help='Input file to prune', required = True)
args = parser.parse_args()
file = args.input

checkpoint = torch.load(file)
new_sd = dict()
for k in checkpoint.keys():
    if k != 'optimizer_states':
        new_sd[k] = checkpoint[k]

save_file(new_sd, f'pruned-{osp.basename(file)}'.replace('.ckpt', '.safetensors'))
# torch.save(new_sd, f'pruned-{osp.basename(file)}')