import torch
import numpy as np
from torch import nn

def synthesize_subtractive(S, hop_length):
    """
    Perform subtractive synthesis by converting frequency domain transfer
    functions into causal, zero-phase, windowed impulse responses
    
    Parameters
    ----------
    S: n_batches x time x n_bands
        Subtractive synthesis parameters
    hop_length: int
        Hop length between subtractive synthesis windows
        
    Returns
    -------
    torch.tensor(n_batches, time*hop_length, 1)
        Subtractive synthesis audio components for each clip
    """
    from utils import fftconvolve
    # Put an imaginary component of all 0s across a new last axis
    # https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
    S = torch.stack([S, torch.zeros_like(S)], -1)
    S = torch.view_as_complex(S)
    # Do the inverse real DFT (assuming symmetry)
    h = torch.fft.irfft(S)
    
    # Shift the impulse response to zero-phase
    nh = h.shape[-1]
    h = torch.roll(h, nh//2, -1)
    # Apply hann window
    h = h*torch.hann_window(nh, dtype=h.dtype, device=h.device)
    # Shift back to causal
    h = nn.functional.pad(h, (0, hop_length-nh))
    h = torch.roll(h, -nh//2, -1)
    
    # Apply the impulse response to random noise in [-1, 1]
    noise = torch.rand(h.shape[0],h.shape[1],hop_length).to(h.device)
    noise = noise*2 - 1
    noise = fftconvolve(noise, h).contiguous()
    
    # Flatten nonoverlapping samples to one contiguous stream
    return noise.reshape(noise.shape[0], noise.shape[1]*noise.shape[2], 1)


def synthesize_additive(A, C, F, S, hop_length, sr, reverb=torch.tensor([])):
    from utils import upsample_time
    from utils import fftconvolve
    AUp = upsample_time(A, hop_length)
    CUp = upsample_time(C, hop_length)
    FUp = upsample_time(F, hop_length)
    FUp = torch.cumsum(FUp, axis=1)/sr
    harmonics = torch.arange(1, C.shape[-1]+1).to(CUp.device)
    harmonics = harmonics.view(1, 1, C.shape[-1])
    FUp = FUp*harmonics
    N = AUp.shape[1]
    Y = AUp*CUp*torch.sin(2*np.pi*FUp)
    Y = torch.sum(Y, axis=-1, keepdims=True)
    YS = synthesize_subtractive(S, hop_length)
    Y = Y + YS
    N = Y.shape[1]
    NR = reverb.detach().cpu().numpy().size
    if NR > 0:
        Y = fftconvolve(Y.squeeze(), reverb.view(1, NR)).unsqueeze(-1)
    return Y
