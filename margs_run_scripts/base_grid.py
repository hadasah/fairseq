import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

CKPT_ROOT_DIR = '/gscratch/zlab/margsli/checkpoint/'
LOG_ROOT_DIR = '/gscratch/zlab/margsli/logs/'

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
parser.add_argument('--baseline', action='store_true', help='Run baseline transformer')
args = parser.parse_args()

gpus = 8
cmd = ('fairseq-train /gscratch/cse/datasets/cc_small/ --task language_modeling '
    '--share-decoder-input-output-embed --sample-break-mode none --ddp-backend=no_c10d '
    '--log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints '
    '--keep-interval-updates 5 --distributed-port 12597 --distributed-world-size {0} '
    '--valid-subset valid').format(gpus)

args_fixed = {}

if args.baseline:
    args_fixed['arch'] = 'transformer_lm'

else:
    args_fixed['arch'] = 'transformer_lm'
    args_fixed['moe-layers'] = 2
    args_fixed['moe-shared-layer'] = ''
    args_fixed['moe-layer-indices'] = '7,8'

if args.baseline:
    name = 'warmup1'
    constraint=None
else:
    name = 'moe30'
    constraint=None

logfolder = 'cc_small/moe/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 0
time_hours = 12
time_minutes = 0

account = 'cse'
partition = 'ckpt'
change_dir = 'fork-fairseq/'
repo = 'fork-fairseq'
exclude = ''

s = gpuscheduler.HyakScheduler(
    verbose=args.verbose, account=account, partition=partition, use_gres=False
)

args_fixed['weight-decay'] = 0.00

fp16 = True
args_sweep = {}

model_dim = 64
ff_dim = 512
moe_ff_dim = 512 
heads = 4

if args.baseline:
    key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim')
    args_sweep[key] = []
    for model_dim in [1024]:
        heads = 8*(model_dim//512)
        for ff_dim in [8192]:
            args_sweep[key].append((model_dim, ff_dim, heads, 0, model_dim, model_dim))
else:
    args_sweep[
        ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'moe-ff-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim')
    ] = [
        (model_dim, ff_dim, moe_ff_dim, heads, 0, model_dim, model_dim)
    ]
    args_sweep['epsilon'] = [0.2]
    args_sweep['criterion'] = ['moe_cross_entropy']
    args_sweep['use-ff-norm'] = [False]
    args_sweep['moe_bloss-weight'] = [0.01]
    args_sweep['moe_bloss_type'] = ['mean']
    args_fixed['special-eval'] = ''

seqs_per_mini_batch = 512 # OpenAI scaling laws mini-batch size

args_sweep[('max-tokens', 'update-freq', 'tokens-per-sample')] = [(2048, 128//gpus, 512)]
args_fixed['validate-interval-updates'] = 1000
args_sweep['decoder-layers'] = [10]
args_sweep[('dropout', 'attention-dropout', 'relu-dropout')] = [(0.0, 0.0, 0.0)] #, (0.1, 0.1, 0.1)]

key = ('lr', 'max-lr', 'min-lr', 'warmup-init-lr')
args_sweep[key] = []
for params in [1e3]:
    lr = 0.003239 + (-0.0001395*math.log(params))
    args_sweep[key].append((lr, lr+1e-8, lr*0.1, lr*0.1 + 1e-8))
args_fixed['lr-scheduler'] = 'cosine'

args_fixed['warmup-updates'] = 3000
args_fixed['fp16-no-flatten-grads'] = ''
args_fixed['min-loss-scale'] = 1e-10
args_sweep['fused'] = [False]
args_sweep['dist-scale'] = [1.00]

args_sweep['prob-quant'] = [False]
args_fixed['optimizer'] = 'adam'
args_sweep['adam-betas'] = ["'(0.9, 0.995)'"]
args_sweep['adam-eps'] = [1e-7]
args_sweep['use-emb-norm'] = [True]
args_sweep[('memory-efficient-fp16', 'adam-bits')] = [(True, 8)]
args_sweep[('clip-norm', 'percentile-clipping')] = [(0.0, 5)]

print(list(args_sweep.keys()))
args4 = []

args5 = {}

# args6 = {}

rdm = np.random.RandomState(5345)

for key, value in args_fixed.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args_prod = []
for key, values in args_sweep.items():
    if isinstance(key, tuple):
        keyvalues = []
        for tups in values:
            arg = ''
            for i, v in enumerate(tups):
                if v is True: v = ''
                if v is False: continue
                if len(key[i]) == 0:
                    arg += '{0} '.format(v)
                else:
                    arg += '--{0} {1} '.format(key[i], v)
            keyvalues.append(arg)
    elif isinstance(key, str):
        keyvalues = []
        for v in values:
            if v is True: v = ''
            if v is False:
                keyvalues.append('')
            else:
                keyvalues.append(' --{0} {1}'.format(key, v))
    args_prod.append(keyvalues)

if len(args_prod) >= 2:
    args_prod = list(product(*args_prod))
else:
    new_args = []
    if len(args_prod) > 0:
        for arg in args_prod[0]:
            new_args.append([arg])
        args_prod = new_args

jobs = []
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            if any([k in job_cmd for k in args5.keys()]):
                for substr, pdict in args5.items():
                    if substr in job_cmd:
                        for key, values in pdict.items():
                            for v in values:
                                job_cmd5 = job_cmd + ' --{0} {1}'.format(key, v)
                                job_cmd5 = job_cmd5 + ' --seed {0}'.format(seed)
                                checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd5).encode('utf-8')).hexdigest(), ckp_name, CKPT_ROOT_DIR)
                                save_dir = ' --save-dir {0}'.format(checkpoint_dir)
                                job_cmd5 = job_cmd5 + save_dir
                                cmds = [job_cmd5]
                                if rdm.rand(1) <= args.p:
                                    jobs.append(job_cmd5)
                                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, CKPT_ROOT_DIR)
                save_dir = ' --save-dir {0}'.format(checkpoint_dir)
                job_cmd = job_cmd + save_dir
                cmds = [job_cmd]
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join(LOG_ROOT_DIR, logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))
else:
    s.run_jobs()

