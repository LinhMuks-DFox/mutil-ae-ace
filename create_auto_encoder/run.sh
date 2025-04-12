#!/usr/bin/env bash

# 该脚本用于在后台启动 AutoEncoderTrainer 训练并输出日志到一个带时间戳的文件。
# 使用方法：给 run.sh 添加可执行权限后，直接在终端运行 ./run.sh

# 若需要使用conda环境，请在此处启用：
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# 定义日志文件名，包含时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="autoencoder_train_${TIMESTAMP}.log"

# 后台运行 make_autoencoder.py，将标准输出和错误输出都重定向到日志文件
nohup python -u create_auto_encoder/make_autoencoder.py > "$LOGFILE" 2>&1 &

echo "AutoEncoder training started."
echo "Logs are being written to $LOGFILE"
echo "Use 'tail -f $LOGFILE' to monitor the log in real time."