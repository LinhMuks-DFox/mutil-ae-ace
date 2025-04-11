#!/bin/bash
if [ -d "Muxkit.DeepLearning.Tools" ]; then
  echo "Muxkit.DeepLearning.Tools already exists. Skipping clone."
else
  git clone git@github.com:LinhMuks-DFox/Muxkit.DeepLearning.Tools.git
  mv Muxkit.DeepLearning.Tools/ MuxkitDeepLearningTools.
fi
if [ -d "MuxKit.AudioSet.IO" ]; then
  echo "MuxKit.AudioSet.IO already exists. Skipping clone."
else
  git clone https://github.com/LinhMuks-DFox/MuxKit.AudioSet.IO
  mv MuxKit.AudioSet.IO/ MuxKitAudioSetIO
fi