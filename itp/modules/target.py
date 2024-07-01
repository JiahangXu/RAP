genai_a100_target_new = \
"""
target:
  service: sing
  name: GenAI-Shared-UKSouth
  vc: singularity-genai-gpu-uksouth
  workspace_name: aims-a100-westus3-WS

environment: 
  image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
  setup:
  - set -x
  image_setup:
  - pip install azureml-sdk
  - pip install accelerate==0.27.2
  - pip install importlib_resources
  - pip install mlflow azureml-mlflow
  - pip list

storage:
  teamdrive:
    storage_account_name: fastnn
    container_name: teamdrive
    mount_dir: /mnt/teamdrive
"""


genai_a100_target = \
"""
target:
  service: sing
  name: aims-sing-res-wus3-02
  vc: singularity_vcs-extended
  workspace_name: aims-a100-westus3-WS

environment: 
  image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
  setup:
  - set -x
  image_setup:
  - pip install azureml-sdk
  - pip install accelerate==0.27.2
  - pip install importlib_resources
  - pip install mlflow azureml-mlflow
  - pip list

storage:
  models:
    storage_account_name: genaimsra
    container_name: models
    mount_dir: /mnt/models
    local_dir: $CONFIG_DIR/fakemodels/
  datasets:
    storage_account_name: genaimsra
    container_name: datasets
    mount_dir: /mnt/datasets
    local_dir: $CONFIG_DIR/fakedatasets/
  logs:
    storage_account_name: genaimsra
    container_name: logs
    mount_dir: /mnt/logs
    local_dir: $CONFIG_DIR/fakelogs/
"""


msrschvc_h100_target = \
"""
target:
  service: aisc
  name: msrresrchvc

environment: 
  image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
  setup:
  - set -x
  image_setup:
  - pip install azureml-sdk
  - pip install accelerate==0.27.2
  - pip install importlib_resources
  - pip install mlflow azureml-mlflow
  - pip list

storage:
  teamdrive:
    storage_account_name: fastnn
    container_name: teamdrive
    mount_dir: /mnt/teamdrive
"""


target_dict = dict(
    genai_a100=genai_a100_target_new,
    msrschvc_h100=msrschvc_h100_target,
)