import datetime
import pathlib
import platform

import torch
import yaml

CacheSize = 2000
Device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
TrainID = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
SaveDirectory = f"./trained/{TrainID}-sound-power"

Platform: str = platform.platform()
DryRun: bool = False
ModelDebugging: bool = True
CompileModel = False

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # 假设你在 src/ 目录下
with open(BASE_DIR / "configs/dataset_info.yml", "r") as f:
    dataset_info = yaml.safe_load(f)

DataSetPath = dataset_info["Dataset"]["5MICFixedESC50"]["data_path"]

TrainSetPath = pathlib.Path(DataSetPath["root"]) / DataSetPath["train"]
TestSetPath = pathlib.Path(DataSetPath["root"]) / DataSetPath["test"]
ValidatePath = pathlib.Path(DataSetPath["root"]) / DataSetPath["validate"]

AutoEncoderHyper = pathlib.Path(BASE_DIR / "configs/auto_encoder_hyperpara.yml")
AutoEncoderCheckPoint = pathlib.Path(BASE_DIR / "autoencoder_checkpoint/run_20250413_155634/checkpoints/epoch_300.pt")
