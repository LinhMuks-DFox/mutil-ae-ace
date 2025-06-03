from . import options

# dataset config
N_Classes = 50  # for esc-50

AudioDuration = 5  # s
WarmpUp = 10
MaxLossOfVisualization = 10
AudioSampleRate = 44100

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

# Train parameters
Epochs = 500 if not options.DryRun else 3
BatchSize = 256 if not options.DryRun else 100
TrainSetTestMileStone = [
    i for i in range(Epochs) if i % 20 == 0 and i != 0
]
SoundTrack = "mix"

# resnet
ResNet = 18

# Light Propagation
Distance = 1
bias = 0.1
std = 0.05
# Camera Response
SignalSourceSampleRate = 15
CameraFrameRate = 30
