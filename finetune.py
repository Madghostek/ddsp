import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader

from dataset import InstrumentDataset
from model import DDSPDecoder
from utils import mss_loss, plot_stft_comparison
from synthesis import synthesize_additive
import tqdm
from pathlib import Path

def do_training(opt):
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)
    
    data = InstrumentDataset(opt.device, opt.dataset_path, opt.ext, opt.sr, opt.hop_length, dataset_size=opt.dataset_size)

    print("dataset size:",len(data.xs))

    ## Step 2: Create model with a test batch
    model = DDSPDecoder(mlp_depth=3, n_units=512, n_harmonics=50, n_bands=65, sr=opt.sr, reverb_len=opt.reverb_len, data=data, hop_length=opt.hop_length)
    model = model.to(opt.device)

    ## Step 3: Setup the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Start at learning rate of 0.001, rate decay of 0.98 factor every 10,000 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000/len(data), gamma=0.98)

    # and we use this to update the parameters
    train_losses = []

    plt.figure(figsize=(12, 12))
    for epoch in range(opt.num_epochs):
        tic = time.time()
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True)
        
        train_loss = 0
        print("loader enum")
        for batch_num, (X, F, FConf, L) in tqdm.tqdm(enumerate(loader)): # Go through each mini batch
            # Move inputs/outputs to GPU
            X = X.to(opt.device)
            F = F.to(opt.device)
            FConf = FConf.to(opt.device)
            L = L.to(opt.device)
            # Reset the optimizer's gradients
            optimizer.zero_grad()
            # Run the model on all inputs
            A, C, P, reverb = model(F, FConf, L)
            # Run the synthesizer
            Y = synthesize_additive(A, C, F, P, opt.hop_length, opt.sr, reverb, use_reverb=opt.impulse)
            # Compute the loss function comparing X to Y
            loss = mss_loss(X, Y)
            # Compute the gradients of the loss function with respect
            # to all of the parameters of the model
            loss.backward()
            # Update the parameters based on the gradient and
            # the optimization scheme
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(loader)
        train_losses.append(train_loss)
        scheduler.step()
            
        # Synthesize training data example
        x = X[0, :, 0].detach().cpu().numpy().flatten()
        y = Y[0, :, 0].detach().cpu().numpy().flatten()
        d = np.array([x, y])
        d = np.array(d*32768, dtype=np.int16)
        wavfile.write(f"{opt.output_path}/Epoch{epoch}.wav", opt.sr, d.T)
        
        plt.clf()
        plot_stft_comparison(F, L, X, Y, reverb, torch.tensor(train_losses))
        plt.tight_layout()
        plt.savefig(f"{opt.output_path}/Epoch{epoch}.png", bbox_inches='tight')
        state = model.state_dict()
        state["loudness_mu"] = model.loudness_mu
        state["loudness_std"] = model.loudness_std
        torch.save(state, f"{opt.output_path}/{epoch+1}.pkl")
        
        print("Epoch {}, loss {:.3f}, elapsed time {:.3f}".format(epoch, train_loss, time.time()-tic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to which to output model")
    parser.add_argument('--device', type=str, default="cuda", help="Torch device to use")
    parser.add_argument('--ext', type=str, default="wav", help="File extention to look for in dataset")
    parser.add_argument('--sr', type=int, default=44100, help="Audio sample rate")
    parser.add_argument('--hop_length', type=int, default=441, help="Hop length to use for pitch/loudness")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=100, help="Batch size for training")
    parser.add_argument('--samples_per_epoch', type=int, default=5000, help="How many random samples to take in the dataset for each epoch")
    parser.add_argument('--reverb_len', type=int, default=44100, help="Length of impulse response to model")

    # finetune args
    parser.add_argument('--initial_model', type=str, required=True, help="Path to initial model")
    parser.add_argument('--dataset_size', type=int, required=False, help="Subset of the dataset to consider")
    parser.add_argument('--impulse', type=bool, default=True, action=argparse.BooleanOptionalAction, help="Allows training ddsp without impulse response which is responsible for echo")
    opt = parser.parse_args()
    if opt.impulse:
        print("Training with impulse (standard)")
    else:
        print("Training without impulse")    
    do_training(opt)

