import gzip
import random
import tqdm
import numpy as np
import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os

from nyx import NYX

import argparse 

from polip import decider

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type = int, default = 256)
parser.add_argument('--model_dim', type = int, default = 512)
parser.add_argument('--layer_depth', type = int, default = 8)
parser.add_argument('--num_attention_heads', type = int, default = 8)
parser.add_argument('--num_batches', type=int, default=int(1e5), help='Number of batches to train')
parser.add_argument('--batch_size', type=int, default=4, help='Size of each training batch')
parser.add_argument('--gradient_accumulate_every', type=int, default=4, help='Steps to accumulate gradients')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimization')
parser.add_argument('--validate_every', type=int, default=100, help='Frequency of validation steps')
parser.add_argument('--prime_length', type=int, default=128, help='Length of the prime sequence for generation')
parser.add_argument('--generate_every', type=int, default=1, help='Frequency of text generation during training')
parser.add_argument('--generate_length', type=int, default=512, help='Length of the generated text')
parser.add_argument('--seq_len', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--sample_prompt', type = str, default = 'AI is a power!')
parser.add_argument('--checkpoint_path', type = str, default = 'weigths/checkpoint_15000.pth')
args = parser.parse_args()

device = decider('cpu')

model = NYX(
    vocab_size=args.vocab_size,  
    model_dim=args.model_dim,        
    layer_depth=args.layer_depth,        
    num_attention_heads=args.num_attention_heads,        
).to(device)

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

optim = Lion(model.parameters(), lr=learning_rate)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    print(f"Checkpoint loaded from {filename}, resuming from step {start_step}")
    return start_step

checkpoint_path = args.checkpoint_path

if os.path.exists(checkpoint_path):
    start_step = load_checkpoint(model, optim, checkpoint_path)
else:
    start_step = 0






sample_prompt = args.sample_prompt
print(f"Sample Prompt: {sample_prompt}")

encoded_prompt = torch.tensor([ord(char) for char in sample_prompt], dtype=torch.long).unsqueeze(0).to(device)
print(f"Encoded Prompt: {encoded_prompt}")

generated_output = model.generate_sequence(generate_length, encoded_prompt)

decoded_output = decode_tokens(generated_output[0].cpu().numpy())


print('Generated Output', '=' * 30)
print(decoded_output, "\n")
