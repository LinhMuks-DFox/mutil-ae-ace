import datetime
import pathlib
import platform
import sys
import torch
import yaml

CacheSize = 2000
Device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
TrainID = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
SaveDirectory = f"./trained/{TrainID}-rpi-ae"

Platform: str = platform.platform()
DryRun: bool = False
ModelDebugging: bool = True

# if linux and torch version > 2, then compile <- Ture
CompileModel = sys.platform.startswith('linux') and int(torch.__version__.split('.')[0]) > 2
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # 假设你在 src/ 目录下
with open(BASE_DIR / "configs/dataset_info.yml", "r") as f:
    dataset_info = yaml.safe_load(f)

DataSetPath = dataset_info["Dataset"]["FixedESC50"]["path"]

TrainSetPath = pathlib.Path(DataSetPath)
TestSetPath = pathlib.Path(DataSetPath)
ValidatePath = pathlib.Path(DataSetPath)

AutoEncoderHyper = pathlib.Path(BASE_DIR / "configs/auto_encoder_hyperpara.yml")
AutoEncoderCheckPoint = pathlib.Path(BASE_DIR / "autoencoder_checkpoint/run_20250413_155634/checkpoints/epoch_300.pt")
