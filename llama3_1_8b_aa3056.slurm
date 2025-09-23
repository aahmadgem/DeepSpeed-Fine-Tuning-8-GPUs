#!/bin/bash
#SBATCH --job-name=llama_100steps
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --gpus=8                    # Number of GPUs per node
#SBATCH --time=01:00:00             # Job time limit
#SBATCH --qos=regular               # QoS setting
#SBATCH --constraint=gpu
#SBATCH --mail-user=aliahmad1331@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=m4431             

# Environment setup
nvidia-smi
source /pscratch/sd/a/aahmad2/DDL/bin/activate

# Set up directory paths and parameters
cd /pscratch/sd/a/aahmad2/DDL/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning

model_name_or_path="/pscratch/sd/a/aahmad2/DDL/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Llama-3.1-8B"
num_train_epochs=1
lora_dim=0
max_seq_len=1024
per_device_train_batch_size=2
current_time=$(date "+%Y%m%d-%H%M%S")

# Define output and stage
OUTPUT=$1
ZERO_STAGE=$2
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
if [ "$OUTPUT" == "" ]; then
   OUTPUT="./output_Llama-3.1-8B_epoch${num_train_epochs}_batch${per_device_train_batch_size}_seq${max_seq_len}_lora${lora_dim}_zero${ZERO_STAGE}_$current_time"
fi
mkdir -p $OUTPUT

# Set up hostfile for multi-node execution
srun hostname > hostfile
sed -i 's/$/ slots=4/' hostfile
cat hostfile

deepspeed --hostfile hostfile main.py \
   --deepspeed_config ds_config.json \
   --sft_only_data_path Dahoas/rm-static \
   --model_name_or_path $model_name_or_path \
   --per_device_train_batch_size $per_device_train_batch_size \
   --per_device_eval_batch_size 4 \
   --max_seq_len $max_seq_len \
   --learning_rate "9.65e-6" \
   --weight_decay 0.0 \
   --num_train_epochs $num_train_epochs  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim $lora_dim \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   --print_loss \
   --save_interval 2000 \
   --enable_tensorboard \
   --tensorboard_path "${OUTPUT}/tensorboard_logs" \
   |& tee $OUTPUT/training.log
