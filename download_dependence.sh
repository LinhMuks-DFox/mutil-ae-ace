#!/bin/bash

mkdir -p lib

# Clone MuxkitTools
if [ -d "lib/MuxkitTools" ]; then
  echo "[✓] MuxkitTools 已存在，跳过克隆。"
else
  git clone git@github.com:LinhMuks-DFox/Muxkit.DeepLearning.Tools.git
  mv Muxkit.DeepLearning.Tools lib/MuxkitTools
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