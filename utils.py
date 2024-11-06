import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import librosa

def upsample_time(X, hop_length, mode='nearest'):
    """
    Upsample a tensor by a factor of hop_length along the time axis
    
    Parameters
    ----------
    X: torch.tensor(M, T, N)
        A tensor in which the time axis is axis 1
    hop_length: int
        Upsample factor
    mode: string
        Mode of interpolation.  'nearest' by default to avoid artifacts
        where notes in the violin jump by large intervals
    
    Returns
    -------
    torch.tensor(M, T*hop_length, N)
        Upsampled tensor
    """
    X = X.permute(0, 2, 1)
    X = nn.functional.interpolate(X, size=hop_length*X.shape[-1], mode=mode)
    return X.permute(0, 2, 1)

def fftconvolve(x, h):
    """
    Perform a fast convolution of two tensors across their last axis
    by using the FFT. Since the DFT assumes circularity, zeropad them 
    appropriately before doing the FFT and slice them down afterwards
    
    The length of the result will be equivalent to np.convolve's 'same'
    
    Parameters
    ----------
    x: torch.tensor(..., N1)
        First tensor
    h: torch.tensor(..., N2)
        Second tensor
    
    Returns
    -------
    torch.tensor(..., max(N1, N2))
    Tensor resulting from the convolution of x and y across their last axis,
    """
    N = max(x.shape[-1], h.shape[-1])
    if x.shape[-1] != h.shape[-1]:
        # Zeropad so they're equal
        if x.shape[-1] < N:
            x = nn.functional.pad(x, (0, N-x.shape[-1]))
        if h.shape[-1] < N:
            h = nn.functional.pad(h, (0, N-h.shape[-1]))
    x = nn.functional.pad(x, (0, N))
    h = nn.functional.pad(h, (0, N))
    X = torch.fft.rfft(x)
    H = torch.fft.rfft(h)
    y = torch.fft.irfft(X*H)
    return y[..., 0:N]


def plot_stft_comparison(F, L, X, Y, reverb, losses=torch.tensor([]), win=1024, sr=16000):
    """
    Some code to help compare the STFTs of ground truth and output audio, while
    also plotting the frequency, loudness, and reverb to get an idea of what the 
    inputs to the network were that gave rise to these ouputs.  It's very helpful
    to call this method while monitoring the training of the network
    
    Parameters
    ----------
    F: torch.tensor(n_batches, n_samples/hop_length, 1)
         Tensor holding the pitch estimates for the clips
    L: torch.tensor(n_batches, n_samples/hop_length, 1)
         Tensor holding the loudness estimates for the clips
    X: torch.tensor(n_batches, n_samples, 1)
        Ground truth audio
    Y: torch.tensor(n_batches, n_samples, 1)
        Output audio from the network->decoder
    reverb: torch.tensor(reverb_len)
        The learned reverb
    losses: list
        A list of losses over epochs over time
    win: int
        Window length to use in the STFT
    sr: int
        Sample rate of audio (used to help make proper units for time and frequency)
    """
    hop = 256
    hann = torch.hann_window(win).to(X)
    SX = torch.abs(torch.stft(X.squeeze(), win, hop, win, hann, return_complex=True))
    SY = torch.abs(torch.stft(Y.squeeze(), win, hop, win, hann, return_complex=True))
    print(SX.shape)
    extent = (0, SX.shape[2]*hop/sr, SX.shape[1]*sr/win, 0)
    plt.subplot(321)
    plt.imshow(torch.log10(SX.detach().cpu()[0, :, :]), aspect='auto', cmap='magma', extent=extent)
    plt.title("Ground Truth")
    plt.ylim([0, 8000])
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (hz)")
    
    plt.subplot(322)
    plt.imshow(torch.log10(SY.detach().cpu()[0, :, :]), aspect='auto', cmap='magma', extent=extent)
    plt.title("Synthesized")
    plt.ylim([0, 8000])
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (hz)")
    
    plt.subplot(323)
    plt.plot(F.detach().cpu()[0, :, 0])
    plt.title("Fundamental Frequency")
    plt.xlabel("Window index")
    plt.ylabel("Hz")
    plt.subplot(324)
    plt.plot(L.detach().cpu()[0, :, 0])
    plt.title("Loudness")
    plt.xlabel("Window Index")
    plt.ylabel("Z-normalized dB")
    if torch.numel(losses) > 0:
        plt.subplot(325)
        plt.plot(losses.detach().cpu().numpy().flatten())
        plt.yscale("log")
        plt.title("Losses (Current {:.3f})".format(losses[-1]))
        plt.xlabel("Epoch")
    plt.subplot(326)
    plt.plot(reverb.detach().cpu().flatten())
    plt.title("Impulse Response")
    plt.xlabel("Sample index")

################################################
# Loudness code modified from original Google Magenta DDSP implementation in tensorflow
# https://github.com/magenta/ddsp/blob/86c7a35f4f2ecf2e9bb45ee7094732b1afcebecd/ddsp/spectral_ops.py#L253
# which, like this repository, is licensed under Apache2 by Google Magenta Group, 2020
# Modifications by Chris Tralie, 2023

def power_to_db(power, ref_db=0.0, range_db=80.0, use_tf=True):
    """Converts power from linear scale to decibels."""
    # Convert to decibels.
    db = 10.0*np.log10(np.maximum(power, 10**(-range_db/10)))
    # Set dynamic range.
    db -= ref_db
    db = np.maximum(db, -range_db)
    return db

def extract_loudness(x, sr, hop_length, n_fft=2048):
    """
    Extract the loudness in dB by using an A-weighting of the power spectrum
    (section B.1 of the paper)

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate (used to figure out frequencies for A-weighting)
    hop_length: int
        Hop length between loudness estimates
    n_fft: int
        Number of samples to use in each window
    """
    # Computed centered STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)
    
    # Compute power spectrogram
    amplitude = np.abs(S)
    power = amplitude**2

    # Perceptual weighting.
    freqs = np.arange(S.shape[0])*sr/n_fft
    a_weighting = librosa.A_weighting(freqs)[:, None]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10**(a_weighting/10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = np.mean(power, axis=0)
    loudness = power_to_db(avg_power)
    return np.array(loudness, dtype=np.float32)

################################################


HANN_TABLE = {}
def mss_loss(X, Y, eps=1e-7):
    loss = 0
    win = 64
    while win <= 2048:
        hop = win//4
        if not win in HANN_TABLE:
            HANN_TABLE[win] = torch.hann_window(win).to(X)
        hann = HANN_TABLE[win]
        SX = torch.abs(torch.stft(X.squeeze(), win, hop, win, hann, return_complex=True))
        SY = torch.abs(torch.stft(Y.squeeze(), win, hop, win, hann, return_complex=True))
        loss_win = torch.sum(torch.abs(SX-SY)) + torch.sum(torch.abs(torch.log(SX+eps)-torch.log(SY+eps)))
        loss += loss_win/torch.numel(SX)
        win *= 2
    return loss