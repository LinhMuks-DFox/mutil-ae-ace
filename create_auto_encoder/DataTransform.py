import torch
import torch.nn as nn
import torchaudio.transforms as T
import yaml


class ToLogMelSpectrogram(nn.Module):
    """
    Convert waveform to Log-Mel Spectrogram using torchaudio.
    """
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.to_db = T.AmplitudeToDB()

    @staticmethod
    def from_yaml(yaml_obj):
        params = yaml_obj.get("ToLogMelSpectrogram", {})
        return ToLogMelSpectrogram(
            sample_rate=params.get("sample_rate", 16000),
            n_fft=params.get("n_fft", 400),
            hop_length=params.get("hop_length", 160),
            n_mels=params.get("n_mels", 80)
        )

    def forward(self, audio_signal: torch.Tensor):
        mel = self.mel_spec(audio_signal)
        return self.to_db(mel)
    
    def __str__(self):
        return (f"ToLogMelSpectrogram(sample_rate={self.sample_rate}, "
                f"n_fft={self.n_fft}, hop_length={self.hop_length}, n_mels={self.n_mels})")


if __name__ == "__main__":
    import torchaudio
    import matplotlib.pyplot as plt

    # Load an example waveform
    import Dataset
    with open("other_configs.yml", "r") as f:
        cfg = yaml.safe_load(f)
    dataset, none = Dataset.AudioSet.from_yaml(cfg, True, False)
    print(dataset[0])
    
    with open("hyperpara.yml", "r") as f:
        hyper = yaml.safe_load(f)
    transformer = ToLogMelSpectrogram.from_yaml(
        hyper
    )
    print(transformer)

    data0 = dataset[0]
    log_mel = transformer(data0)
    # # Visualize
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel[0].numpy(), aspect="auto", origin="lower")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig("log_mel")