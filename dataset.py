import numpy as np
import torch
import os
import tqdm
import pesto
from torch.utils.data import Dataset


def walk_dir(dir:str, ext="wav"):
    """
    Return an array listing the paths to each .wav file in the 
    given directory and any subdirectories
    Parameters:
        dir: str of the directory to walk through

    Returns:
        [strings]: array of strings, each representing a unique file path
                in the given directory that ends with .wav
    """
    files = []
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            if str(f).endswith(f".{ext}"):
                files.append(f)
        if os.path.isdir(f):
            append_this = walk_dir(f)
            for each in append_this:
                files.append(each)
    return files

class InstrumentDataset(Dataset):
    def __init__(self, folder, ext, sr, hop_length, 
                 sample_len=3, samples_per_epoch=5000):
        """
        Parameters
        ----------
        folder: string
            Path to folder
        ext: string
            Extension of files to look for in folder
        sr: int
            Sample rate
        hop_length: int
            Hop length between frequency and loudness frames
        sample_len: float
            Length, in seconds, of each sample
        samples_per_epoch: int
            What to consider the length of this data
        """
        from utils import extract_loudness
        import librosa
        self.timesteps = int(sample_len*sr/hop_length)
        self.samples_per_epoch = samples_per_epoch
        self.hop_length = hop_length

        self.loudnesses = []
        self.pitches = []
        self.confidences = []
        self.xs = []
        files = walk_dir(folder, ext)
        for audio_filename in tqdm.tqdm(files):
            x, self.sr = librosa.load(audio_filename, sr=sr)
            x = x/np.max(np.abs(x))
            
            ## Step 1: Compute loudness
            loudness = extract_loudness(x, sr, hop_length)
            
            ## Step 2: Compute pitch
            _, pitch, confidence, _ = pesto.predict(torch.from_numpy(x), sr, step_size=1000*hop_length/sr)
            
            ## Step 3: Crop all aspects to be the same
            N = min(loudness.size, pitch.shape[0])
            if N >= self.timesteps:
                # Only add if the clip is long enough
                self.loudnesses.append(loudness[0:N])
                self.pitches.append(pitch[0:N])
                self.confidences.append(confidence[0:N])
                self.xs.append(x[0:N*hop_length])
            else:
                tqdm.tqdm.write(f"Skipping {audio_filename}: too short at {N*hop_length/sr} seconds")
        
        ## Step 4: Normalize loudnesses
        all_loudnesses = np.concatenate(self.loudnesses)
        self.loudness_mu = np.mean(all_loudnesses)
        self.loudness_std = np.std(all_loudnesses)
        for i, loudness in enumerate(self.loudnesses):
            self.loudnesses[i] = (loudness-self.loudness_mu)/self.loudness_std
        
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Return a tuple (x, pitch, loudness)
        """
        # First extract a random tune
        idx = np.random.randint(len(self.loudnesses))
        x = self.xs[idx]
        pitch = self.pitches[idx]
        confidence = self.confidences[idx]
        loudness = self.loudnesses[idx]

        # Next extract random index into latent variables
        idx = np.random.randint(len(pitch)-self.timesteps)
        # Extract and scale audio
        x = x[idx*self.hop_length:(idx+self.timesteps)*self.hop_length]
        x = torch.from_numpy(x).view(x.size, 1)
        # Extract pitch and loudness
        pitch = pitch[idx:idx+self.timesteps].view(self.timesteps, 1)
        confidence = confidence[idx:idx+self.timesteps].view(self.timesteps, 1)
        loudness = torch.from_numpy(loudness[idx:idx+self.timesteps]).view(self.timesteps, 1)
        return x, pitch, confidence, loudness
        
