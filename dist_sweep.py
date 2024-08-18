# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import os
import copy
import submitit

from train import main, parse_args


def run_sweep(main, opt):
    # main = glue_main

    log_folder = 'training_logs'
    # num_workers = getattr(opt, "num_workers", 1)
    # ngpus_per_node = getattr(opt, "ngpus_per_node", 1)
    num_workers = 24
    ngpus_per_node = 1

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=2400,
                               slurm_partition="a100",
                               slurm_comment='',
                               slurm_nodes=8, # opt.nodes 1 for now
                               slurm_ntasks_per_node=1, # distributed not supported
                               slurm_cpus_per_task=num_workers,
                               slurm_gpus_per_node=ngpus_per_node, # distributed not supported
                               #slurm_array_parallelism=20,
    )

    jobs = []
    def _launch(opt, executor, jobs):
        job = executor.submit(main, opt)
        jobs.append(job)

    with executor.batch():
        _launch(opt, executor, jobs)

    print('launched:')
    for job in jobs:
        print(job.job_id)


if __name__ == "__main__":
    args, args_text = parse_args()
    run_sweep(main, (args, args_text))
