import datetime
import platform
import pathlib
import torch

CacheSize = 2000
Device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
TrainID = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
SaveDirectory = f"./trained/{TrainID}-resnet"

Platform: str = platform.platform()
DryRun: bool = False
ModelDebugging: bool = True
CompileModel = False
DataSetPath = pathlib.Path(__file__).parent.parent / "esc50_data/ESC-50-master/"

