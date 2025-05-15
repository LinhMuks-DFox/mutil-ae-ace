from . import options

# dataset config
N_Classes = 50  # for esc-50
N_RASPI = 2

ResetRoomInterval = 10


AudioDuration = 5  # s
WarmpUp = 10
MaxLossOfVisualization = 10

AudioSampleRate = 44100
DownsampleRate = 16000
# Optimizer & Scheduler
LearningRate = 0.001
WeightDecay = 5e-5

ReduceLROnPlateauMode, ReduceLROnPlateauFactor, ReduceLROnPlateauPatience, ReduceLROnPlateauMinLR = 'min', 0.5, 10, 1e-7
ReduceLROnPlateauMetricsName = "validate_loss"

T_0, T_mult, eta_min = 10, 1, 1e-5
SchedulerParameter = {
    "gamma": 0.1,
    "milestones": [40, 80, ]
}
ROOM_DIMS_MIN = [3, 5, 7]
ROOM_DIMS_MAX = [4, 6, 8]
ABSORPTION_MIN = 0.001
ABSORPTION_MAX = 0.01
MAX_ORDER_MIN = 1
MAX_ORDER_MAX = 3
DISTANCE_SOURCE_WALL_MIN = 1
SOURCE_POSITION_METHOD = ""

SHIFT_MAX_RATIO = 0.5
SCALE_MIN = 0.01
SCALE_MAX = 1.0

RoomMaterial = {
    "ceiling": "hard_surface",
    "floor": "brickwork",
    "east": "brickwork",
    "west": "brickwork",
    "north": "brickwork",
    "south": "brickwork",
}
# Train parameters
Epochs = 300 if not options.DryRun else 3
BatchSize = 100 if not options.DryRun else 100
TrainSetTestMileStone = [
    i for i in range(Epochs) if i % 20 == 0 and i != 0
]
SoundTrack = "mix"

# resnet
ResNet = 18
