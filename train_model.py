"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""
import math
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav


from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

import torch
import torch.optim as optim
import torch.nn as nn

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

semantic_path = "C:\Users\Jaswanth Kakani\OneDrive\Desktop\saikumar\sunobark\suno\bark\semantic_output_pytorch.bin" 
coarse_path = "C:\Users\Jaswanth Kakani\OneDrive\Desktop\saikumar\sunobark\suno\bark\coarse_pytorch_model.bin" 
fine_path = "C:\Users\Jaswanth Kakani\OneDrive\Desktop\saikumar\sunobark\suno\bark\pytorch_model.bin"

preload_models(
    text_use_gpu=True,
    text_use_small=False,
    text_model_path=semantic_path,
    coarse_use_gpu=True,
    coarse_use_small=False,
    coarse_model_path=coarse_path,
    fine_use_gpu=True,
    fine_use_small=False,
    fine_model_path=fine_path,
    codec_use_gpu=True,
    force_reload=False,
    path="suno"
)
audio_array = generate_with_settings(
    text_prompt,
    semantic_temp=0.7,
    semantic_top_k=50,
    semantic_top_p=0.99,
    coarse_temp=0.7,
    coarse_top_k=50,
    coarse_top_p=0.95,
    fine_temp=0.5,
    voice_name="C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\wavs_harsha",
    use_semantic_history_prompt=False,
    use_coarse_history_prompt=True,
    use_fine_history_prompt=True,
    output_full=False
)

write_wav(filepath, SAMPLE_RATE, audio_array)
def generate_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):
    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name if use_fine_history_prompt else None,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen,
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        FULL_T = k.shape[-2]

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if past_kv is not None:
                # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
                # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
                # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so 
                # to work around this we set is_causal=False.
                is_causal = False
            else:
                is_causal = True

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,FULL_T-T:FULL_T,:FULL_T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.input_vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        if past_kv is not None:
            assert t == 1
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        else:
            if merge_context:
                assert(idx.shape[1] >= 256+256+1)
                t = idx.shape[1] - 256
            else:
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the GPT model itself
            if merge_context:
                tok_emb = torch.cat([
                    self.transformer.wte(idx[:,:256]) + self.transformer.wte(idx[:,256:256+256]),
                    self.transformer.wte(idx[:,256+256:])
                ], dim=1)
            else:
                tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_kv[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # shape (1, t)
            assert position_ids.shape == (1, t)

        pos_emb = self.transformer.wpe(position_ids) # position embeddings of shape (1, t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.transformer.h, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv = new_kv + (kv,)

        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return (logits, new_kv)
    


num_samples = 1000 
sequence_length = 128  
batch_size = 32 
data = torch.randint(0, 10048, (num_samples, sequence_length), dtype=torch.long)
targets = torch.randint(0, 10048, (num_samples, sequence_length), dtype=torch.long)

# Create a DataLoader
dataset = TensorDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the learning rate
learning_rate = 0.001  # You can set the learning rate to your desired value

config = GPTConfig(
    block_size=1024,
    input_vocab_size=10048,  # Set to the appropriate vocabulary size
    output_vocab_size=10048,  # Set to the appropriate vocabulary size
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,  # Set your desired dropout rate
    bias=True
)
model = GPT(config) 



# Step 1: Text Processing
class TextTokenizer:
    def __init__(self):
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
        self.tokenizer.add_tokens('[PAD]')

    def tokenize_text(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

tokenizer = TextTokenizer()
class WaveNetBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation)
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)

    def forward(self, x):
        dilated_output = torch.tanh(self.dilated_conv(x))
        input_padded = nn.functional.pad(x, (self.dilated_conv.padding[0], 0))
        input_gate = torch.sigmoid(self.dilated_conv(input_padded))
        gated_output = dilated_output * input_gate

        skip_output = self.skip_conv(gated_output)
        residual_output = self.residual_conv(gated_output) + x[:, :, -residual_channels:]

        return skip_output, residual_output

class WaveNet(nn.Module):
    def __init__(self, num_blocks, residual_channels, dilation_channels, skip_channels, output_channels, kernel_size):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([WaveNetBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2 ** i) for i in range(num_blocks)])
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.final_conv2 = nn.Conv1d(skip_channels, output_channels, 1)

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            skip_output, residual_output = block(x)
            skip_connections.append(skip_output)
            x = residual_output

        x = torch.sum(torch.stack(skip_connections, dim=0), dim=0)
        x = torch.relu(self.final_conv1(x))
        x = self.final_conv2(x)
        return x

class WaveNetVocoder:
    def __init__(self, num_blocks, residual_channels, dilation_channels, skip_channels, output_channels, kernel_size):
        self.wavenet = WaveNet(num_blocks, residual_channels, dilation_channels, skip_channels, output_channels, kernel_size)

    def synthesize_audio(self, features):
        with torch.no_grad():
            predicted_waveform = self.wavenet(features.unsqueeze(0)).squeeze(0)
        return predicted_waveform

# Step 2: Dataset and DataLoader
class VoiceCloningDataset(Dataset):
    def __init__(self, text_file_path, audio_data_dir, sample_rate=16000, max_waveform_length=160000):
        with open(text_file_path, 'r') as file:
            self.text_data = file.readlines()

        self.audio_data = []
        self.mfcc_data = []
        self.sample_rate = sample_rate
        self.max_waveform_length = max_waveform_length

        for i in range(len(self.text_data)):
            audio_file_path = os.path.join(audio_data_dir, f'audio_{i}.wav')
            try:
                print(f"Loading audio file: {audio_file_path}")
                waveform, _ = torchaudio.load(audio_file_path, num_frames=-1, normalize=True)
                
                # Pad or trim the audio waveform to the desired length
                target_length = self.max_waveform_length
                padded_waveforms = pad_sequence(audio_waveform, batch_first=True, padding_value=0)
                if len(waveform) < target_length:
                    pad_amount = target_length - len(waveform)
                    waveform = F.pad(waveform, (0, pad_amount))
                elif len(waveform) > target_length:
                    waveform = waveform[:, :target_length]

                # Extract MFCC features using a function like compute_mfcc_features
                mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate)
                mfcc_features = mfcc_transform(waveform)

                self.audio_data.append(waveform)
                self.mfcc_data.append(torch.tensor(mfcc_features))
            except Exception as e:
                print(f"Error loading audio file {audio_file_path}: {e}")

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        try:
            if idx < len(self.text_data):
                text = self.text_data[idx]
                audio_waveform = self.audio_data[idx]

                # Extract MFCC features for the audio waveform
                mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate)
                mfcc_features = mfcc_transform(audio_waveform)

                return text, audio_waveform, torch.tensor(mfcc_features)
            else:
                # Handle the case when the index is out of range
                return None
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            return None
# Step 3: Model Definition
class VoiceCloningSystem(nn.Module):
    def __init__(self, text_embedding_dim, num_mfcc_features, hidden_size):
        super(VoiceCloningSystem, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.num_mfcc_features = num_mfcc_features
        self.hidden_size = hidden_size

        # Define your model architecture
        self.text_embedding_layer = nn.Embedding(len(tokenizer.tokenizer), text_embedding_dim)
        self.audio_processing_layer = nn.LSTM(input_size=num_mfcc_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(text_embedding_dim + hidden_size, num_mfcc_features)

    def forward(self, text_tokens, audio_features):
        text_embedded = self.text_embedding_layer(text_tokens)
        audio_hidden, _ = self.audio_processing_layer(audio_features)
        combined_features = torch.cat((text_embedded, audio_hidden[:, -1, :]), dim=1)
        output_features = self.output_layer(combined_features)
        return output_features

# Step 4: Training Loop
text_file_path = "C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\list_har - Copy.txt"
audio_data_dir = "C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\wavs_harsha"


# Define hyperparameters
text_embedding_dim = 128
num_mfcc_features = 13
hidden_size = 256
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Create dataset and data loader
dataset = VoiceCloningDataset(text_file_path, audio_data_dir)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = VoiceCloningSystem(text_embedding_dim, num_mfcc_features, hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (text_tokens, audio_waveform, mfcc_features) in enumerate(train_loader):
        optimizer.zero_grad()
        predicted_mfcc = model(text_tokens, audio_waveform)
        loss = criterion(predicted_mfcc, mfcc_features)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
torch.save(model.state_dict(), 'voice_cloning_model.pth')
print("Trained model saved.")
