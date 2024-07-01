import re
import datetime
import subprocess
import random
import string
import os
import argparse
from modules.target import target_dict
template = \
"""
description: {job_name}

{target}

code:
  local_dir: $CONFIG_DIR/../../

jobs:
{jobs}
"""

job_template = \
"""- name: {job_name}
  sku: G8
  priority: high
  mpi: True

  command:
    - pip install vllm transformers fire
    - sudo apt-get update
    - sudo apt-get install git-lfs
    - git lfs install
    - cd /scratch
    # - git clone {model_url}
    # - cd /scratch/amlt_code/
    # - {script}
    - sleep infinity

  submit_args: 
    env:
      {{DEBUG: 1}}
"""

model_url = {
    # masked
    "Meta-Llama-3-8B": None
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='genai_a100', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument("--mark", type=str, required=False, default=None)
    parser.add_argument("--script", type=str, required=True, default=None)
    
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    # find model_ckpt and task in "--model_ckpt ../Llama-2-7b-hf --data_split test_all --task folio"
    model_name = re.search(r'--model_ckpt\s+(\S+)', args.script).group(1).split("/")[-1]
    task = re.search(r'--task\s+(\S+)', args.script).group(1)

    date = datetime.datetime.now().strftime('%m%d%H%M')
    job_name = f'{args.mark}-{task}-{model_name}-{date}'
    jobs = job_template.format(
        job_name=job_name,
        model_url=model_url[model_name],
        model_name=model_name,
        script=args.script,
        debug=mode,
    )
    description = f'{job_name}'

    # ======================================================================================================
    # Don't need to modify following code
    result = template.format(
        job_name=job_name,
        jobs=jobs,
        target=target_dict[args.target], 
    )   
    print(result)

    tmp_name = ''.join(random.choices(string.ascii_lowercase, k=6)) + job_name
    tmp_name = os.path.join("./.tmp", tmp_name)
    with open(tmp_name, "w") as fout:
        fout.write(result)
    if mode == 0:
        subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    else:
        # subprocess.run(f'amlt run -d {description} {tmp_name} {job_name}', shell=True)
        subprocess.run(["amlt", "run", "-d", description, tmp_name, job_name])

if __name__ == "__main__":
    main()
