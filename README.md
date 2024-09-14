# Natural Yielding of eXpressions (NYX) Model

<img src="assets/logo.png" alt="Sample Image" width="800" height = "400"/>



**NYX** is a Transformer-based neural architecture designed for autoregressive natural language generation tasks. The model leverages positional rotary embeddings, multi-head self-attention, and low-rank adaptation techniques for efficient, scalable training. The architecture is implemented using PyTorch and provides modular flexibility to suit a variety of NLP applications, including text generation, completion, and language modeling.




## Model Overview

The NYX model is structured as a deep Transformer neural network with the following primary characteristics:

- **Model Parameters**: 30,020,096 total trainable parameters.
- **Architecture**: Incorporates 8 Transformer layers, each consisting of multi-head self-attention, feedforward networks, and residual connections.
- **Positional Encoding**: Rotary embeddings with XPOS scaling for improved attention mechanism efficiency in sequence modeling.
- **Low-Rank Fine-Tuning**: Supports LoRA modules to reduce memory footprint and computational overhead during fine-tuning.
- **Regularization**: Includes dropout layers in both the attention and feedforward components for increased robustness against overfitting.

The model operates with a vocabulary size of 50257 tokens and can be trained for autoregressive tasks using causal masking, ensuring that each token prediction is conditioned solely on preceding tokens.

## Features

- **Parameter Efficiency**: The NYX model, despite its relatively compact size (30M parameters), delivers high performance across a range of NLP tasks thanks to its careful balance of model dimensionality and advanced mechanisms like rotary embeddings and LoRA.
- **Rotary Embeddings**: Used for positional encoding, rotary embeddings ensure that the attention mechanism captures relative positions within the sequence, enabling the model to generalize better across varying sequence lengths.
- **LoRA Fine-Tuning**: Fine-tuning is implemented using low-rank adapters, which decompose the weight matrices into low-rank components, reducing the overall training cost while maintaining model performance.
- **Causal Masking**: Designed for autoregressive modeling, ensuring that each token prediction only depends on previously generated tokens.

## Model Architecture

The NYX model consists of the following key components:

1. **Embedding Layer**: An embedding matrix maps input tokens to a high-dimensional space of size 128. This embedding is shared with the output layer to improve weight sharing and reduce parameter redundancy.
2. **Transformer Layers**: The model contains 2 Transformer layers, each composed of:
    - **Multi-Head Attention**: Each layer has 4 heads, with each head having a dimensionality of 32. The attention mechanism includes rotary embeddings for efficient positional encoding.
    - **Feedforward Networks**: A feedforward network with a scaling factor of 2x the model dimension.
    - **Residual Connections**: For improved gradient flow and model stability, the output of the attention and feedforward layers is combined with the input via residual connections.
3. **Output Layer**: A linear layer projects the hidden states back to the vocabulary size, enabling token prediction.

## Installation

Ensure you have the following dependencies:

- Python 3.8+
- PyTorch 1.8+
- Lion Optimizer (for optimization)
- Beartype (for type checking)
- Einops (for tensor operations)
- Accelerate (for distributed training)

To install these packages, run:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, you can use the provided \`train.py\` script. Run the following command to start training:

```bash
python train.py --filepath <DATA FILE PATH> --vocab_size 256 --model_dim 512 --layer_depth 8 --num_attention_heads 8 --num_batches 100000 --batch_size 4 --gradient_accumulate_every 4 --learning_rate 1e-4 --validate_every 100 --prime_length 128 --generate_every 500 --generate_length 512 --seq_len 1024
```

Replace the \`<DATA FILE PATH>\` with the path to the dataset you want to use for training.

## Testing the Model

In order to test the model, you first need to download trained weights from the provided link (e.g., [link]). After this, move the weights to a folder named \`./weights\`. Then, run the following command to generate text based on a prompt:

```bash
python test.py --sample_prompt "AI is a power!" --vocab_size 256 --model_dim 512 --layer_depth 8 --num_attention_heads 8 --generate_length 512
```

This command will generate text based on the sample prompt "AI is a power!" with the specified model configuration.

## Checkpoints

To resume training from a checkpoint, you can load saved weights using the \`train.py\` script. Ensure that you have the checkpoint file in the \`weights\` folder, then modify the training script to load the checkpoint:

```bash
python train.py --filepath <DATA FILE PATH> --checkpoint_path weights/checkpoint_15000.pth
```





