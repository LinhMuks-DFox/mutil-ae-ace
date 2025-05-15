import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import numpy as np
import typing

# Assuming pyroomacoustics is installed and importable
import pyroomacoustics as pra
from torchaudio import transforms as Tడియో # Alias to avoid conflict if user has other T

from src.AutoEncoder import AutoEncoder
from src.LatentDataset import RIRWaveformToMelTransform, LatentTransform
from lib.MuxkitTools.audio_tools.PyroomSimulation import RoomSimulation, random_microphone_array_position

from . import options as opt
from . import hyperparameters as hyp

class AdjustForResNet(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # This reshape is very specific. Ensure it matches downstream model's exact needs.
        # Input: [B, N_MIC, LATENT_DIM]
        # Output example for some ResNet: [B, 1, N_MIC * k, LATENT_DIM // k]
        # The original "*4" and "//4" implies LATENT_DIM should be divisible by 4.
        # For now, this module is NOT called by DataPreprocessor.forward if output is [B, N_MIC, LATENT_DIM]
        if x.dim() == 3:
            batch, n_mic, features = x.shape
            # Example: if you want to treat N_MIC as height and LATENT_DIM as width for a 2D conv
            # and need to create a "channel" dimension of 1.
            # return x.unsqueeze(1) # -> [B, 1, N_MIC, LATENT_DIM]
            # The original reshape was:
            if features % 4 == 0 and n_mic > 0 :
                 return x.reshape(batch, 1, n_mic * 4, features // 4).contiguous()
            else:
                # Fallback or error if specific reshape is not possible
                # print(f"AdjustForResNet: Cannot apply specific reshape. Returning as [B, 1, N_MIC, LATENT_DIM]")
                return x.unsqueeze(1) # A common way to make it [B, C, H, W] like for ResNet
        elif x.dim() == 2: # For a single instance [N_MIC, LATENT_DIM]
            n_mic, features = x.shape
            if features % 4 == 0 and n_mic > 0:
                return x.reshape(1, 1, n_mic * 4, features // 4).contiguous()
            else:
                return x.unsqueeze(0).unsqueeze(0)
        return x

def _generate_random_room_parameters():
    """Generates randomized parameters for a pyroomacoustics room."""
    room_dim = [random.uniform(hyp.ROOM_DIMS_MIN[i], hyp.ROOM_DIMS_MAX[i]) for i in range(3)]
    absorption = random.uniform(hyp.ABSORPTION_MIN, hyp.ABSORPTION_MAX)
    max_order = random.randint(hyp.MAX_ORDER_MIN, hyp.MAX_ORDER_MAX)

    # Source position
    if hyp.SOURCE_POSITION_METHOD == "center":
        source_pos = [dim / 2 for dim in room_dim]
    else: # random
        # Ensure source is not too close to walls
        min_dist = getattr(hyp, "DISTANCE_SOURCE_WALL_MIN", 0.1)
        source_pos = [random.uniform(min_dist, dim - min_dist) for dim in room_dim]
    
    # Microphone positions
    # Ensure mics are not too close to walls
    mic_min_dist_wall = getattr(hyp, "DISTANCE_MIC_WALL_MIN", 0.1)
    # random_microphone_array_position expects space_size relative to origin 0
    # so if we want mics to be min_dist away from walls, we effectively reduce the available space
    mic_space_size = [dim - 2 * mic_min_dist_wall for dim in room_dim]
    for i in range(len(mic_space_size)): # Ensure space is not negative
        if mic_space_size[i] <=0: mic_space_size[i] = room_dim[i]*0.1 # fallback to small portion if constraint too tight
            
    mic_positions_relative = random_microphone_array_position(mic_space_size, hyp.N_MIC)
    # Shift mic positions to be relative to room origin + min_dist_wall
    mic_positions = mic_positions_relative + np.array([[mic_min_dist_wall]] * len(room_dim))


    return {
        "room_dim": room_dim,
        "absorption": absorption,
        "max_order": max_order,
        "source_pos": source_pos,
        "mic_positions": mic_positions,
        "fs": hyp.RESAMPLE_RATE # pyroomacoustics room needs sample rate
    }

def make_new_pra_room():
    """Callable that returns a new pyroomacoustics.Room with random parameters."""
    params = _generate_random_room_parameters()
    room = pra.ShoeBox(
        params["room_dim"],
        fs=params["fs"],
        absorption=params["absorption"],
        max_order=params["max_order"]
    )
    room.add_source(params["source_pos"])
    room.add_microphone_array(params["mic_positions"])
    return room

class RandomRIR(nn.Module):
    def __init__(self, n_mic: int, reset_interval: int, device: str = "cpu"):
        super().__init__()
        self.n_mic = n_mic
        self.device = device # RoomSimulation will move RIRs to its device
        self.reset_interval = reset_interval
        self.call_count = 0
        self.current_room_sim = None
        self._create_new_room_simulation() # Initial room

    def _create_new_room_simulation(self):
        # make_new_pra_room will be called inside RoomSimulation to create the room
        self.current_room_sim = RoomSimulation(make_new_pra_room, self.n_mic).to(self.device)
        # print(f"RandomRIR: New room simulation created. RIR shape: {self.current_room_sim.rir.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: tensor representing a single audio sample, 
        # expected shape [1, 1, time_axis] (batch_size=1, num_channels_in=1, time_length)
        # after resampling.
        # Output: [1, n_mic, time_axis_convolved]
        
        if self.training: # Only change rooms during training
            self.call_count += 1
            if self.call_count >= self.reset_interval:
                self._create_new_room_simulation()
                self.call_count = 0
        
        if x.dim() == 1: # Input is [time_axis]
            # Add batch and channel dimension for consistency
            single_audio_input_for_roomsim = x.unsqueeze(0) # -> [1, time_axis]
        elif x.dim() == 2: # Input is [1, time_axis] (channel, time) or [batch_size=1, time_axis]
            if x.shape[0] == 1: # Assumed to be [1, time_axis] (channel, time)
                single_audio_input_for_roomsim = x
            else: # Should not happen if called from __getitem__ with batch_size=1 for DataPreprocessor
                raise ValueError(f"RandomRIR expects a single audio item, but got batch dim {x.shape[0]} in 2D input.")
        elif x.dim() == 3: # Input is [1, 1, time_axis] (batch, channel, time)
            if x.shape[0] == 1 and x.shape[1] == 1:
                single_audio_input_for_roomsim = x.squeeze(0) # -> [1, time_axis]
            else:
                raise ValueError(f"RandomRIR expects input shape [1,1,T], got {x.shape}")
        else:
            raise ValueError(f"RandomRIR received unexpected input dimensions: {x.dim()}. Expected 1D, 2D [1,T] or 3D [1,1,T].")

        # RoomSimulation expects [num_channels_in, time_axis]
        # single_audio_input_for_roomsim is now [1, time_axis]
        convolved_audio = self.current_room_sim(single_audio_input_for_roomsim) # Output: [n_mic, time_convolved]
        
        # Add back the batch dimension for consistency with subsequent modules
        return convolved_audio.unsqueeze(0) # -> [1, n_mic, time_convolved]


class RandomShift(nn.Module):
    def __init__(self, max_shift_ratio: float = 0.1):
        super().__init__()
        self.max_shift_ratio = max_shift_ratio
        if not (0 <= max_shift_ratio <= 1):
            raise ValueError("max_shift_ratio must be between 0 and 1.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [batch_size, num_mics, time_axis]
        # Output: [batch_size, num_mics, time_axis] (with shifts and padding)
        batch_size, num_mics, time_axis = x.shape
        output_tensor = torch.zeros_like(x)
        max_shift_samples = int(time_axis * self.max_shift_ratio)

        for b in range(batch_size): # This loop is fine as DataPreprocessor.forward will pass B=1
            for m in range(num_mics):
                shift_samples = random.randint(-max_shift_samples, max_shift_samples)
                current_channel_audio = x[b, m, :]
                
                if shift_samples == 0:
                    output_tensor[b, m, :] = current_channel_audio
                elif shift_samples > 0: 
                    len_to_keep = time_axis - shift_samples
                    if len_to_keep > 0:
                        output_tensor[b, m, shift_samples:] = current_channel_audio[:len_to_keep]
                else: 
                    abs_shift = -shift_samples
                    len_to_keep = time_axis - abs_shift
                    if len_to_keep > 0:
                         output_tensor[b, m, :len_to_keep] = current_channel_audio[abs_shift:]
        return output_tensor

class RandomScale(nn.Module):
    def __init__(self, scale_min: float, scale_max: float):
        super().__init__()
        if scale_min > scale_max:
            raise ValueError("scale_min cannot be greater than scale_max.")
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [batch_size, num_mics, time_axis]
        # Output: [batch_size, num_mics, time_axis] (scaled)
        batch_size, num_mics, _ = x.shape # This loop is fine as DataPreprocessor.forward will pass B=1
        scales = (self.scale_min + (self.scale_max - self.scale_min) * torch.rand(batch_size, num_mics, 1, device=x.device))
        return x * scales


class DataPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        
        current_device = torch.device(opt.Device if torch.cuda.is_available() or opt.Device=="cpu" else "cpu") # Ensure valid device

        with open(opt.AutoEncoderHyper, "r") as f:
            hyper_ae_config = yaml.safe_load(f)
            
            self.auto_encoder = AutoEncoder.from_structure_hyper_and_checkpoint(
                hyper_ae_config["AutoEncoder"], 
                opt.AutoEncoderCheckPoint.resolve(),
                device=current_device
            ).eval() # Set to eval mode as it's a pre-trained feature extractor

            self.mel_transform_pipeline = RIRWaveformToMelTransform.from_hyper(hyper_ae_config, device=current_device)

        self.rand_rir = RandomRIR(n_mic=hyp.N_MIC, 
                                  reset_interval=hyp.RESET_ROOM_INTERVAL,
                                  device=current_device)
        self.rand_shift = RandomShift(max_shift_ratio=hyp.SHIFT_MAX_RATIO)
        self.rand_scale = RandomScale(scale_min=hyp.SCALE_MIN, scale_max=hyp.SCALE_MAX)
        
        self.ae_latent_dim = hyper_ae_config["AutoEncoder"]["latent_size"]


    @torch.no_grad() # Preprocessing should not require gradients
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: raw audio from Dataset.__getitem__, typically shape [1 (channel), time_samples_at_orig_sr]
                 e.g., for a single item: [1, 220500] for 5s @ 44.1kHz.
                 The DataLoader will later batch these to [B, 1, T_orig].
                 This forward method processes one item, effectively B=1.
        Output: latents, shape [1, N_MIC, ae_latent_dim] (for a single input item)
        """
        # Ensure input has a batch-like dimension for initial transforms if it's [C,T]
        if x.dim() == 2: # Expected [1, T_orig] from dataset __getitem__
            x = x.unsqueeze(0) # -> [1, 1, T_orig] to act as a batch of 1
        elif x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1:
            # If called from elsewhere with an actual batch, this preprocessor might need adjustments
            # or this check should be more robust. For __getitem__, [1,1,T] is expected after unsqueeze.
            raise ValueError(f"DataPreprocessor expects input from __getitem__ as [1,C,T] or [C,T], received {x.shape}")

        # 0. Resample to target SR (e.g., 44.1kHz -> 16kHz)
        # Input to resample: [1, 1, T_orig]
        # Output of resample: [1, 1, T_resampled]
        x_resampled = self.mel_transform_pipeline.resample(x)

        # 1. Apply random room RIR
        # Input to rand_rir: [1, 1, T_resampled]
        # Output of rand_rir: [1, N_MIC, T_convolved]
        x_rir = self.rand_rir(x_resampled)

        # 2. Randomly shift each channel (simulates asynchronicity)
        # Input: [1, N_MIC, T_convolved]
        # Output: [1, N_MIC, T_convolved] (length maintained by padding)
        x_shifted = self.rand_shift(x_rir)

        # 3. Apply random scale (simulates attenuation/gain)
        # Input: [1, N_MIC, T_convolved]
        # Output: [1, N_MIC, T_convolved]
        x_scaled = self.rand_scale(x_shifted)

        # 4. Fix time-length to target duration (e.g., 5s at RESAMPLE_RATE)
        # Input: [1, N_MIC, T_convolved]
        # Output: [1, N_MIC, T_fixed_16k_5s]
        x_fixed_length = self.mel_transform_pipeline.time_fixer(x_scaled)
        
        # 5. Convert to multi-channel Mel Spectrograms
        batch_size_eff, num_mics, time_fixed = x_fixed_length.shape # batch_size_eff will be 1
        
        x_for_melspec = x_fixed_length.reshape(batch_size_eff * num_mics, 1, time_fixed)
        mel_specs = self.mel_transform_pipeline.melspec(x_for_melspec)
        log_mel_specs = self.mel_transform_pipeline.to_db(mel_specs)
        log_mel_squeezed = log_mel_specs.squeeze(1) # -> [1*N_MIC, N_MELS, T_mel_frames]

        # 6. Apply AutoEncoder's encoder
        # Input: [1*N_MIC, N_MELS, T_mel_frames]
        # Output: [1*N_MIC, ae_latent_dim]
        latents_flat = self.auto_encoder.encode(log_mel_squeezed)
        
        # 7. Reshape to final output for this single item: [1, N_MIC, ae_latent_dim]
        # This shape [1, N_MIC, ae_latent_dim] will then be collated by DataLoader to [B, N_MIC, ae_latent_dim]
        output_latents = latents_flat.reshape(batch_size_eff, num_mics, self.ae_latent_dim)

        return output_latents
