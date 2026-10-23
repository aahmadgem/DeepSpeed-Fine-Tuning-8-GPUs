# Distributed Fine Tuning of Llama 3.1 8B with DeepSpeed

Course project implementing distributed supervised fine tuning of **Llama 3.1 8B** using **DeepSpeed Chat** on an HPC cluster.

## Project Overview

The goal of this project was to configure and run large language model fine tuning across multiple GPUs and compute nodes using DeepSpeed.

The training workflow used:

- **2 compute nodes**
- **4 GPUs per node**
- **8 GPUs total**
- **DeepSpeed ZeRO Stage 3**
- **100 training steps**
- **Gradient checkpointing**
- **TensorBoard logging**

The training used the `Dahoas/rm-static` dataset with Llama 3.1 8B as the base model.

## Training Configuration

| Parameter | Value |
| --- | --- |
| Base model | Llama 3.1 8B |
| Dataset | Dahoas/rm-static |
| Framework | DeepSpeed Chat |
| Training type | Supervised Fine Tuning |
| Compute nodes | 2 |
| GPUs | 8 total |
| GPUs per node | 4 |
| ZeRO stage | 3 |
| Training steps | 100 |
| Batch size | 2 per GPU |
| Maximum sequence length | 1024 |
| Learning rate | 9.65e-6 |
| Scheduler | Cosine |
| Gradient accumulation | 1 |
| Gradient checkpointing | Enabled |
| Seed | 1234 |

## Distributed Training Workflow

Training was submitted through:

`llama3_1_8b_aa3056.slurm`

The SLURM job launched the DeepSpeed Chat supervised fine tuning entry point across the two compute nodes.

A simplified version of the training command is:

```bash
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

`main.py` is part of the DeepSpeed Chat implementation used for the assignment rather than a separate `train.py` included in this repository.

## Training Results

The distributed run completed **100 training steps**.

Average loss decreased from approximately **5.3** at the beginning of training to approximately **2.0** near step 100.

![Average loss over training steps](figures/average_loss_vs_steps.png)

Most of the loss reduction occurred between approximately steps 20 and 45, after which training stabilized near an average loss of 2.

The corresponding aggregated loss decreased from approximately **43** to approximately **16**.

[View aggregated loss plot](figures/aggregate_loss_vs_steps.png)

## Repository Contents

### `llama3_1_8b_aa3056.slurm`

SLURM job configuration used to launch the distributed DeepSpeed training run across two compute nodes.

### `training.log`

Console output from the distributed training run, including the reported training loss over the 100 steps.

### `main_ckpt.py`

Modified DeepSpeed Chat training code used in the course assignment for checkpoint related functionality.

### `utils.py`

Supporting utility modifications used with the assignment.

### `figures/`

Loss curves generated from the training output.

## Training Outcome

This project demonstrates:

- distributed LLM training across multiple compute nodes and GPUs
- DeepSpeed ZeRO Stage 3 configuration
- SLURM based HPC job execution
- gradient checkpointing for memory efficient training
- training metric extraction and convergence analysis
- practical debugging of a distributed training workflow

The loss curve shows substantial convergence over the required 100 training steps.

## Note on Model Saving

The training process reached all 100 required steps.

During the final tokenizer saving stage, the run encountered a `NotImplementedError` associated with `save_vocabulary()`.

The distributed training and loss reporting completed before this serialization issue.
