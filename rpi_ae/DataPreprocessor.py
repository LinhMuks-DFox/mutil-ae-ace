import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import numpy as np
import typing

import pyroomacoustics as pra
from torchaudio import transforms as Tడియో

from src.AutoEncoder import AutoEncoder
from src.LatentDataset import RIRWaveformToMelTransform, LatentTransform
from lib.MuxkitTools.audio_tools.PyroomSimulation import RoomSimulation, random_microphone_array_position

from . import options as opt
from . import hyperparameters as hyp

# --- Utility: AdjustForResNet (remains the same, but placement decision later) ---
class AdjustForResNet(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.dim() == 3: # Batch, N_MIC, latent_dim
            batch, n_mic, features = x.shape
            if features % 4 == 0 and n_mic > 0 :
                 return x.reshape(batch, 1, n_mic * 4, features // 4).contiguous()
            else:
                return x.unsqueeze(1) 
        elif x.dim() == 2: # N_MIC, latent_dim (no batch)
            n_mic, features = x.shape
            if features % 4 == 0 and n_mic > 0:
                return x.reshape(1, 1, n_mic * 4, features // 4).contiguous()
            else:
                return x.unsqueeze(0).unsqueeze(0)
        return x

# --- Room Generation Utilities (remains the same) ---
def _generate_random_room_parameters():
    room_dim = [random.uniform(hyp.ROOM_DIMS_MIN[i], hyp.ROOM_DIMS_MAX[i]) for i in range(3)]
    absorption = random.uniform(hyp.ABSORPTION_MIN, hyp.ABSORPTION_MAX)
    max_order = random.randint(hyp.MAX_ORDER_MIN, hyp.MAX_ORDER_MAX)
    if hyp.SOURCE_POSITION_METHOD == "center":
        source_pos = [dim / 2 for dim in room_dim]
    else:
        min_dist = getattr(hyp, "DISTANCE_SOURCE_WALL_MIN", 0.1)
        source_pos = [random.uniform(min_dist, dim - min_dist) for dim in room_dim]
    
    mic_min_dist_wall = getattr(hyp, "DISTANCE_MIC_WALL_MIN", 0.1)
    mic_space_size = [dim - 2 * mic_min_dist_wall for dim in room_dim]
    for i in range(len(mic_space_size)):
        if mic_space_size[i] <=0: mic_space_size[i] = room_dim[i]*0.1
            
    mic_positions_relative = random_microphone_array_position(mic_space_size, hyp.N_MIC)
    mic_positions = mic_positions_relative + np.array([[mic_min_dist_wall]] * len(room_dim))
    return {"room_dim": room_dim, "absorption": absorption, "max_order": max_order,
            "source_pos": source_pos, "mic_positions": mic_positions, "fs": hyp.RESAMPLE_RATE}

def make_new_pra_room():
    params = _generate_random_room_parameters()
    room = pra.ShoeBox(params["room_dim"], fs=params["fs"], absorption=params["absorption"], max_order=params["max_order"])
    room.add_source(params["source_pos"])
    room.add_microphone_array(params["mic_positions"])
    return room

# --- Augmented Preprocessor for Training ---
class AugmentedDataPreprocessor(nn.Module): # Renamed from RandomRIR to avoid confusion
    def __init__(self, n_mic: int, reset_interval: int, device: str = "cpu"):
        super().__init__()
        self.n_mic = n_mic
        self.device = device
        self.reset_interval = reset_interval
        self.call_count = 0
        self.current_room_sim = None
        self._create_new_room_simulation()

    def _create_new_room_simulation(self):
        self.current_room_sim = RoomSimulation(make_new_pra_room, self.n_mic).to(self.device)

    def forward(self, x: torch.Tensor, is_training: bool) -> torch.Tensor: # Added is_training flag
        if is_training: # Only change rooms during training
            self.call_count += 1
            if self.call_count >= self.reset_interval:
                self._create_new_room_simulation()
                self.call_count = 0
        
        single_audio_input_for_roomsim = None
        if x.dim() == 1: 
            single_audio_input_for_roomsim = x.unsqueeze(0)
        elif x.dim() == 2 and x.shape[0] == 1:
            single_audio_input_for_roomsim = x
        elif x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
            single_audio_input_for_roomsim = x.squeeze(0)
        else:
            raise ValueError(f"AugmentedDataPreprocessor (RIR) received unexpected input dimensions: {x.shape}.")

        convolved_audio = self.current_room_sim(single_audio_input_for_roomsim)
        return convolved_audio.unsqueeze(0)

class RandomShift(nn.Module): # (Implementation remains the same as rpi_ae_preprocessor_v2)
    def __init__(self, max_shift_ratio: float = 0.1):
        super().__init__()
        self.max_shift_ratio = max_shift_ratio
        if not (0 <= max_shift_ratio <= 1):
            raise ValueError("max_shift_ratio must be between 0 and 1.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_mics, time_axis = x.shape
        output_tensor = torch.zeros_like(x)
        max_shift_samples = int(time_axis * self.max_shift_ratio)
        for b in range(batch_size):
            for m in range(num_mics):
                shift_samples = random.randint(-max_shift_samples, max_shift_samples)
                current_channel_audio = x[b, m, :]
                if shift_samples == 0: output_tensor[b, m, :] = current_channel_audio
                elif shift_samples > 0: 
                    len_to_keep = time_axis - shift_samples
                    if len_to_keep > 0: output_tensor[b, m, shift_samples:] = current_channel_audio[:len_to_keep]
                else: 
                    abs_shift = -shift_samples
                    len_to_keep = time_axis - abs_shift
                    if len_to_keep > 0: output_tensor[b, m, :len_to_keep] = current_channel_audio[abs_shift:]
        return output_tensor

class RandomScale(nn.Module): # (Implementation remains the same as rpi_ae_preprocessor_v2)
    def __init__(self, scale_min: float, scale_max: float):
        super().__init__()
        if scale_min > scale_max: raise ValueError("scale_min cannot be greater than scale_max.")
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_mics, _ = x.shape
        scales = (self.scale_min + (self.scale_max - self.scale_min) * torch.rand(batch_size, num_mics, 1, device=x.device))
        return x * scales

# --- Main DataPreprocessor (for Training) ---
class RPIAE_Train_DataPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        current_device = torch.device(opt.Device if torch.cuda.is_available() or opt.Device=="cpu" else "cpu")
        with open(opt.AutoEncoderHyper, "r") as f:
            hyper_ae_config = yaml.safe_load(f)
            self.auto_encoder = AutoEncoder.from_structure_hyper_and_checkpoint(
                hyper_ae_config["AutoEncoder"], opt.AutoEncoderCheckPoint.resolve(), device=current_device
            ).eval()
            self.mel_transform_pipeline = RIRWaveformToMelTransform.from_hyper(hyper_ae_config, device=current_device)
        
        self.rand_rir_augmenter = AugmentedDataPreprocessor(
            n_mic=hyp.N_MIC, 
            reset_interval=hyp.RESET_ROOM_INTERVAL, # This will be BatchSize as per discussion
            device=current_device
        )
        self.rand_shift = RandomShift(max_shift_ratio=hyp.SHIFT_MAX_RATIO)
        self.rand_scale = RandomScale(scale_min=hyp.SCALE_MIN, scale_max=hyp.SCALE_MAX)
        self.ae_latent_dim = hyper_ae_config["AutoEncoder"]["latent_size"]
        self.is_training = True # Default to training mode for this preprocessor

    def train(self, mode: bool = True):
        super().train(mode)
        self.is_training = mode
        return self

    def eval(self):
        super().eval()
        self.is_training = False
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: x = x.unsqueeze(0) 
        if not (x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1):
             raise ValueError(f"RPIAE_Train_DataPreprocessor expects input from __getitem__ as [1,1,T] or [1,T], received {x.shape}")

        x_resampled = self.mel_transform_pipeline.resample(x)
        x_rir = self.rand_rir_augmenter(x_resampled, self.is_training) # Pass training status
        x_shifted = self.rand_shift(x_rir)
        x_scaled = self.rand_scale(x_shifted)
        x_fixed_length = self.mel_transform_pipeline.time_fixer(x_scaled)
        
        batch_size_eff, num_mics, time_fixed = x_fixed_length.shape
        x_for_melspec = x_fixed_length.reshape(batch_size_eff * num_mics, 1, time_fixed)
        mel_specs = self.mel_transform_pipeline.melspec(x_for_melspec)
        log_mel_specs = self.mel_transform_pipeline.to_db(mel_specs)
        log_mel_squeezed = log_mel_specs.squeeze(1)
        
        latents_flat = self.auto_encoder.encode(log_mel_squeezed)
        output_latents = latents_flat.reshape(batch_size_eff, num_mics, self.ae_latent_dim)
        return output_latents

# --- Clean DataPreprocessor (for Validation/Test) ---
class RPIAE_Eval_DataPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        current_device = torch.device(opt.Device if torch.cuda.is_available() or opt.Device=="cpu" else "cpu")
        with open(opt.AutoEncoderHyper, "r") as f:
            hyper_ae_config = yaml.safe_load(f)
            self.auto_encoder = AutoEncoder.from_structure_hyper_and_checkpoint(
                hyper_ae_config["AutoEncoder"], opt.AutoEncoderCheckPoint.resolve(), device=current_device
            ).eval()
            self.mel_transform_pipeline = RIRWaveformToMelTransform.from_hyper(hyper_ae_config, device=current_device)
        
        self.ae_latent_dim = hyper_ae_config["AutoEncoder"]["latent_size"]
        # For evaluation, N_MIC from hyperparameters is still relevant if we need to match shape by repeating
        self.n_mic_out = hyp.N_MIC 

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: clean raw audio from Dataset.__getitem__, shape [1 (channel), time_samples_at_orig_sr]
        Output: latents, shape [1, N_MIC, ae_latent_dim] (single channel latent repeated N_MIC times)
        """
        if x.dim() == 2: x = x.unsqueeze(0)
        if not (x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1):
             raise ValueError(f"RPIAE_Eval_DataPreprocessor expects input from __getitem__ as [1,1,T] or [1,T], received {x.shape}")

        # 0. Resample
        x_resampled = self.mel_transform_pipeline.resample(x) # Output: [1, 1, T_resampled]

        # No RIR, No RandomShift, No RandomScale for clean evaluation

        # 4. Fix time-length
        # time_fixer expects [B, C, T]. Input x_resampled is [1, 1, T_resampled]
        x_fixed_length = self.mel_transform_pipeline.time_fixer(x_resampled) # Output: [1, 1, T_fixed]
        
        # 5. Convert to Mel Spectrogram
        # melspec and to_db expect [B*C, 1, T]. Here B=1, C=1.
        # x_fixed_length is already [1, 1, T_fixed]
        mel_spec = self.mel_transform_pipeline.melspec(x_fixed_length) # Output: [1, 1, N_MELS, T_mel]
        log_mel_spec = self.mel_transform_pipeline.to_db(mel_spec)    # Output: [1, 1, N_MELS, T_mel]
        log_mel_squeezed = log_mel_spec.squeeze(1) # -> [1, N_MELS, T_mel]
        
        # 6. Apply AutoEncoder's encoder
        # Input: [1, N_MELS, T_mel]
        # Output: [1, ae_latent_dim]
        latent_single_mic = self.auto_encoder.encode(log_mel_squeezed)
        
        # 7. Repeat latent features N_MIC times to match training output shape for the model
        # Output: [1, N_MIC, ae_latent_dim]
        output_latents = latent_single_mic.unsqueeze(1).repeat(1, self.n_mic_out, 1)
        
        return output_latents
