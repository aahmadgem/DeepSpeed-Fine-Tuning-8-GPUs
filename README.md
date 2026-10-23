# Distributed Llama 3.1 8B Fine Tuning on an HPC Cluster

This project demonstrates distributed large language model training using **DeepSpeed**, **SLURM**, and multi node GPU computing.

I configured and executed supervised fine tuning of **Llama 3.1 8B** across **8 GPUs on 2 compute nodes**, using DeepSpeed ZeRO Stage 3 to manage model memory and distributed execution.

## Why This Project

Training large language models introduces several practical engineering challenges:

- distributing computation across multiple GPUs and nodes
- coordinating jobs through an HPC scheduler
- managing model memory efficiently
- configuring reproducible training runs
- tracking convergence and diagnosing failures in distributed environments

This project focused on building and running that workflow end to end.

## HPC and Distributed Training Setup

| Component | Configuration |
| --- | --- |
| Base model | Llama 3.1 8B |
| Dataset | Dahoas/rm-static |
| Framework | DeepSpeed Chat |
| Compute nodes | 2 |
| GPUs per node | 4 |
| Total GPUs | 8 |
| Parallel training | DeepSpeed |
| ZeRO optimization | Stage 3 |
| Batch size | 2 per GPU |
| Maximum sequence length | 1024 |
| Learning rate | 9.65e-6 |
| Scheduler | Cosine |
| Gradient accumulation | 1 |
| Gradient checkpointing | Enabled |
| Training interval | 100 steps |
| Seed | 1234 |

## Workflow

The training job was submitted through:

`llama3_1_8b_aa3056.slurm`

The SLURM script coordinated the two compute nodes and launched the DeepSpeed distributed training process through the DeepSpeed Chat supervised fine tuning entry point.

A simplified version of the launch command is:

```bash id="k5x2tm"
deepspeed --hostfile hostfile main.py \
  --sft_only_data_path Dahoas/rm-static \
  --model_name_or_path Llama-3.1-8B \
  --per_device_train_batch_size 2 \
  --max_seq_len 1024 \
  --learning_rate 9.65e-6 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --seed 1234 \
  --gradient_checkpointing \
  --zero_stage 3 \
  --deepspeed
```

The `main.py` entry point belongs to the DeepSpeed Chat implementation used for the training workflow rather than a separate training script created in this repository.

## Training Convergence

The distributed run reached the required **100 training steps**.

Average training loss decreased from approximately **5.3** at the beginning of the run to approximately **2.0** near step 100.

![Average loss over training steps](figures/average_loss_vs_steps.png)

The largest reduction occurred during the early and middle stages of training, followed by stabilization near an average loss of 2.

The corresponding aggregated loss decreased from approximately **43** to approximately **16**.

[View aggregated loss plot](figures/aggregate_loss_vs_steps.png)

## What I Worked On

My contribution to the project included:

- configuring the multi node SLURM training job
- launching DeepSpeed across 8 GPUs
- working with ZeRO Stage 3 distributed optimization
- configuring training hyperparameters and gradient checkpointing
- monitoring distributed training behavior
- extracting and visualizing training loss
- diagnosing the final checkpoint serialization error
- documenting the workflow for reproducibility

## Repository Structure

```text id="u6jzq1"
.
├── README.md
├── llama3_1_8b_aa3056.slurm
├── training.log
├── main_ckpt.py
├── utils.py
├── DeepSpeed_Dahos-rm env.txt
└── figures/
    ├── average_loss_vs_steps.png
    └── aggregate_loss_vs_steps.png
```

### `llama3_1_8b_aa3056.slurm`

SLURM configuration used to allocate the HPC resources and launch distributed DeepSpeed training.

### `training.log`

Console output from the distributed run, including reported training loss and runtime diagnostics.

### `main_ckpt.py`

Modified DeepSpeed Chat training code used for checkpoint related functionality.

### `utils.py`

Supporting utility modifications used in the training workflow.

### `figures/`

Training convergence plots generated from the recorded loss values.

## Engineering Takeaways

This project provided hands on experience with the practical infrastructure behind large scale AI training rather than only model development.

Key areas included:

- multi node GPU orchestration
- HPC job scheduling
- distributed optimization
- memory efficient training
- experiment monitoring
- reproducible execution
- debugging distributed workflows

These same workflow principles are applicable to other large scale scientific computing and machine learning workloads where model execution must be distributed across shared HPC resources.

## Known Issue

The training process reached all 100 required steps.

During final tokenizer saving, the workflow encountered a `NotImplementedError` associated with `save_vocabulary()`.

The distributed training and loss reporting completed before this serialization issue.
