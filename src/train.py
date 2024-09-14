import gzip
import random
import tqdm
import numpy as np

from nyx import NYX

import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type = str, default = 'data/enwik8')
parser.add_argument('--vocab_size', type=int, default=256)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--layer_depth', type=int, default=8)
parser.add_argument('--num_attention_heads', type=int, default=8)
parser.add_argument('--num_batches', type=int, default=int(1e5), help='Number of batches to train')
parser.add_argument('--batch_size', type=int, default=4, help='Size of each training batch')
parser.add_argument('--gradient_accumulate_every', type=int, default=4, help='Steps to accumulate gradients')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimization')
parser.add_argument('--validate_every', type=int, default=100, help='Frequency of validation steps')
parser.add_argument('--prime_length', type=int, default=128, help='Length of the prime sequence for generation')
parser.add_argument('--generate_every', type=int, default=1, help='Frequency of text generation during training')
parser.add_argument('--generate_length', type=int, default=512, help='Length of the generated text')
parser.add_argument('--seq_len', type=int, default=1024, help='Maximum sequence length')
args = parser.parse_args()

num_batches = args.num_batches
batch_size = args.batch_size
gradient_accumulate_every = args.gradient_accumulate_every
learning_rate = args.learning_rate
validate_every = args.validate_every
prime_length = args.prime_length
generate_every = args.generate_every
generate_length = args.generate_length
seq_len = args.seq_len

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

accelerator = Accelerator()
device = accelerator.device

model = NYX(
    vocab_size=args.vocab_size,  
    model_dim=args.model_dim,        
    layer_depth=args.layer_depth,        
    num_attention_heads=args.num_attention_heads,        
).to(device)

file_path = args.filepath

with open(file_path, 'rb') as file:
    data = np.frombuffer(file.read(), dtype=np.uint8).copy()

np_train, np_valid = np.split(data, [int(90e6)])
data_train, data_val = torch.from_numpy(np_train).to(device), torch.from_numpy(np_valid).to(device)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, seq_len)
val_dataset = TextSamplerDataset(data_val, seq_len)
train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size))
val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size))

optim = Lion(model.parameters(), lr=learning_rate)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

for i in tqdm.tqdm(range(num_batches), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(gradient_accumulate_every):
        loss = model(next(train_loader).to(device), compute_loss = True)
        accelerator.backward(loss / gradient_accumulate_every)

    accelerator.print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % validate_every == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader).to(device), compute_loss = True)
            accelerator.print(f"validation loss: {loss.item()}")

    if i % generate_every == 0:
        model.eval()
        inp = random.choice(val_dataset)[:prime_length].to(device)
        prime = decode_tokens(inp.cpu().numpy())
        accelerator.print(f"{prime} \n\n {'*' * 100}")

        sample = model.generate_sequence(generate_length, inp[None, ...])
        output_str = decode_tokens(sample[0].cpu().numpy())
        accelerator.print(output_str, "\n")

