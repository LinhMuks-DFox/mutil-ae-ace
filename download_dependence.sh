#!/bin/bash

mkdir -p lib
if [-d lib/__init__.py] then
  echo "lib/__init__.py 已经存在，跳过创建"
else
  touch lib/__init__.py
fi
# Define target path
TARGET_DIR="lib/MuxkitTools"
REPO_URL="git@github.com:LinhMuks-DFox/Muxkit.DeepLearning.Tools.git"
TAG_VERSION="v1.0.0"

if [ -d "$TARGET_DIR" ]; then
  echo "[✓] MuxkitTools 已存在，跳过克隆。"
else
  echo "[->] 正在克隆 MuxkitTools $TAG_VERSION ..."
  
  # -b $TAG_VERSION : 指定克隆 v1.0.0 这个标签
  # --depth 1       : 浅克隆，只下载该标签的最新状态，不下载历史记录（下载极快）
  # 最后一个参数      : 直接克隆到 lib/MuxkitTools 目录，省去了 mv 操作
  git clone --depth 1 -b "$TAG_VERSION" "$REPO_URL" "$TARGET_DIR"
  
  if [ $? -eq 0 ]; then
    echo "[✓] MuxkitTools $TAG_VERSION 安装完成。"
  else
    echo "[x] 克隆失败，请检查网络或 SSH Key 配置。"
    exit 1
  fi
fi

# Clone AudioSet IO
if [ -d "lib/AudioSet" ]; then
  echo "[✓] AudioSet 已存在，跳过克隆。"
else
  git clone https://github.com/LinhMuks-DFox/MuxKit.AudioSet.IO.git
  mv MuxKit.AudioSet.IO/AudioSet lib/AudioSet
  rm -rf MuxKit.AudioSet.IO
fi

# Clone ESC50 IO
if [ -d "lib/esc50_io" ]; then
  echo "[✓] esc50_io 已存在，跳过克隆。"
else
  git clone git@github.com:LinhMuks-DFox/esc-50-io.git
  mv esc-50-io lib/esc50_io
fi