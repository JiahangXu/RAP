import torch
import time
import argparse

import subprocess
import re
import random

def get_gpu_util(rank):

    cmds = [
        'nvidia-smi',
        '-i',
        str(rank),
    ]
    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    outputs = stdout.decode('utf-8').split('\n')
    
    util = 0
    for output in outputs[::-1]:
        # switch to performance line
        if 'Default' in output:
            # match all the numbers and return the last one
            util = re.findall(r'\d+', output)[-1]
            util = int(util)
            break
    else:
        print("rank {}: couldn't match any, check GPU status!".format(rank))
    return util

def keep(rank, args):
    
    torch.cuda.set_device(rank)

    print(f'benchmarking {args.gpus} gpus...')
    while True:
        # Randomly generate n between 5 and 9
        n = random.randint(8, 10)
        
        # Generate tensors a and b with dimensions 8192*n, 8192
        a = torch.rand((8192 * n, 8192)).cuda()
        b = torch.rand((8192 * n, 8192)).cuda()

        tic = time.time()
        for _ in range(100000):
            c = a * b
        torch.cuda.synchronize()
        toc = time.time()
        # if rank == 0:
        #     print('benchmark 8K matmul: time span: {}ms'.format((toc - tic) * 1000 / 5000))
        time.sleep(args.interval)
        while True:
            util = get_gpu_util(rank)
            if util <= 10:
                break
            # print('rank {}: find gpu busy, keep sleeping...'.format(rank))
            time.sleep(args.interval)
        # print('rank {} gets up'.format(rank))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    torch.multiprocessing.spawn(keep, args=(args,), nprocs=args.gpus, join=True)
