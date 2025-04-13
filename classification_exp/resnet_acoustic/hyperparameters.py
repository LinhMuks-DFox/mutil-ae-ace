from . import options

# dataset config
N_Classes = 50  # for esc-50
DataSampleRate = 44100
AudioDuration = 5  # s

# Blinkies configuration
NewSampleRate = 44100

# Optimizer & Scheduler
LearningRate = 0.1
T_0, T_mult, eta_min = 10, 1, 1e-5
SchedulerParameter = {
    "gamma": 0.1,
    "milestones": [40, 80, ]
}

# Train parameters
Epochs = 100 if not options.DryRun else 3
BatchSize = 100 if not options.DryRun else 100
TrainSetTestMileStone = [
    i for i in range(Epochs) if i % 20 == 0 and i != 0
]

# Data Preprocessing
SoundTrack = "mix"

# resnet
ResNet = 18
