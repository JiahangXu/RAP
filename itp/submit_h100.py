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
  local_dir: $CONFIG_DIR/../

jobs:
{jobs}
"""

job_template = \
"""- name: {job_name}
  sku: NDH100v5:G8
  priority: high
  mpi: True

  command:
    # - pip install -U datasets
    - pip install vllm==0.4.0.post1 transformers==4.39.3
    - python dynamic_ft.py
    - sleep infinity

  submit_args: 
    env:
      {{DEBUG: 1}}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='msrschvc_h100', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument("--mark", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    date = datetime.datetime.now().strftime('%m%d%H%M')
    job_name = f'{args.mark}-{date}'
    jobs = job_template.format(
        job_name=job_name, 
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
