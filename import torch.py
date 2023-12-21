import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")
    

import sys
sys.executable

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import soundfile as sf
from tqdm import tqdm
from transformers import BertTokenizer
from dataclasses import dataclass
from typing import Optional
import numpy as np
from coqpit import Coqpit
from encodec import EncodecModel
from transformers import BertTokenizer
from TTS.tts.layers.bark.inference_funcs import (
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    generate_voice,
    load_voice,
)
from TTS.tts.layers.bark.load_model import load_model
from TTS.tts.layers.bark.model import GPT
from TTS.tts.layers.bark.model_fine import FineGPT
from TTS.tts.models.base_tts import BaseTTS

# Install required libraries
from transformers import BertTokenizer
from dataclasses import dataclass
from coqpit import Coqpit
from TTS.tts.models.base_tts import BaseTTS
from typing import Optional
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


@dataclass
class FineConfig:
    num_layers: int = 6
    embedding_size: int = 512
    vocab_size: int = 30000
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
@dataclass
class CoarseConfig:
    num_layers: int = 6
    embedding_size: int = 512
    vocab_size: int = 30000
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
@dataclass
class SemanticConfig:
    num_layers: int = 6
    embedding_size: int = 512
    vocab_size: int = 30000
    num_heads: int = 8 
    ff_dim: int = 2048
    dropout: float = 0.1
# Define BarkAudioConfig
@dataclass
class BarkAudioConfig(Coqpit):
    num_chars: int = 256
    semantic_config: SemanticConfig = field(default_factory=SemanticConfig)
    coarse_config: CoarseConfig = field(default_factory=CoarseConfig)
    fine_config: FineConfig = field(default_factory=FineConfig)
    sample_rate: int = 24000
    output_sample_rate: int = 24000
    def __init__(self, num_chars=0, model_args=None):
            super().__init__()
            self.num_chars = num_chars
            self.model_args = model_args if model_args is not None else {}
            self.semantic_config = SemanticConfig()
            self.coarse_config = CoarseConfig()
            self.fine_config = FineConfig()
class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()

        # Multi-Head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout)

        # Feedforward Neural Network
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_size)
        )

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self Attention
        attention_output, _ = self.attention(x, x, x)

        # Residual Connection and Layer Normalization
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)

        # Feedforward Neural Network
        ff_output = self.feedforward(x)

        # Residual Connection and Layer Normalization
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self._init_weights(self.embedding.weight)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(config.embedding_size, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)])

        # Linear output layer
        self.output_layer = nn.Linear(config.embedding_size, config.vocab_size)
        self._init_weights(self.output_layer.weight)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_data):
        embedded_input = self.embedding(input_data)

        transformer_output = embedded_input
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        output_data = self.output_layer(transformer_output)

        return output_data
class FineGPT(nn.Module):
    def __init__(self, config):
        super(FineGPT, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self._init_weights(self.embedding.weight)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(config.embedding_size, config.num_heads, config.ff_dim, config.dropout) for _ in range(config.num_layers)])

        # Linear output layer
        self.output_layer = nn.Linear(config.embedding_size, config.vocab_size)
        self._init_weights(self.output_layer.weight)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_data):
        embedded_input = self.embedding(input_data)

        transformer_output = embedded_input
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        output_data = self.output_layer(transformer_output)

        return output_data

class EncodecModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(EncodecModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Recurrent layers (LSTM)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, 128)

    def forward(self, x):
        # Input shape: (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Reshape for recurrent layers
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, channels)

        # Recurrent layers
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(lstm_out)

        return output

    def set_target_bandwidth(self, bandwidth):
            # Adjust the model parameters based on the target bandwidth
            if bandwidth < 6.0:
                # If the target bandwidth is less than 6.0, reduce the number of LSTM layers
                self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True)

    @staticmethod
    def encodec_model_24khz():
        # Implement a method to create and return an instance of your EncodecModel
        return EncodecModel()

# Instantiate the EncodecModel
encodec_model = EncodecModel.encodec_model_24khz()

# Set target bandwidth (example value, implement based on your requirements)
encodec_model.set_target_bandwidth(6.0)

# Define Bark TTS class
class Bark(BaseTTS):
    def __init__(
        self,
        config: Coqpit,
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    ) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.config.num_chars = len(tokenizer)
        self.tokenizer = tokenizer
        self.semantic_model = GPT(config.semantic_config)
        self.coarse_model = GPT(config.coarse_config)
        self.fine_model = FineGPT(config.fine_config)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)
    def _set_model_args(self, config):
            config_num_chars = (
                config["num_chars"] if isinstance(config, dict) and "num_chars" in config else config.num_chars
            )
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars

    @property
    def device(self):
        return next(self.parameters()).device

    def load_bark_models(self):
        self.semantic_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["text"], device=self.device, config=self.config, model_type="text"
        )
        self.coarse_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["coarse"],
            device=self.device,
            config=self.config,
            model_type="coarse",
        )
        self.fine_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["fine"], device=self.device, config=self.config, model_type="fine"
        )

    def train_step(self):
        pass

    def text_to_semantic(self, text: str, history_prompt: Optional[str] = None, temp: float = 0.7, base=None, allow_early_stop=True, **kwargs):
        x_semantic = generate_text_semantic(
            text,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        return x_semantic

    def semantic_to_waveform(self, semantic_tokens: np.ndarray, history_prompt: Optional[str] = None, temp: float = 0.7, base=None):
        x_coarse_gen = generate_coarse(
            semantic_tokens,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self,
            history_prompt=history_prompt,
            temp=0.5,
            base=base,
        )
        audio_arr = codec_decode(x_fine_gen, self)
        return audio_arr, x_coarse_gen, x_fine_gen

    def generate_audio(self, text: str, history_prompt: Optional[str] = None, text_temp: float = 0.7, waveform_temp: float = 0.7, base=None, allow_early_stop=True, **kwargs):
        x_semantic = self.text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        audio_arr, c, f = self.semantic_to_waveform(
            x_semantic, history_prompt=history_prompt, temp=waveform_temp, base=base
        )
        return audio_arr, [x_semantic, c, f]

    def generate_voice(self, audio, speaker_id, voice_dir):
        if voice_dir is not None:
            voice_dirs = [voice_dir]
            try:
                _ = load_voice(speaker_id, voice_dirs)
            except (KeyError, FileNotFoundError):
                output_path = os.path.join(voice_dir, speaker_id + ".npz")
                os.makedirs(voice_dir, exist_ok=True)
                generate_voice(audio, self, output_path)

    def _set_voice_dirs(self, voice_dirs):
        def_voice_dir = None
        if isinstance(self.config.DEF_SPEAKER_DIR, str):
            os.makedirs(self.config.DEF_SPEAKER_DIR, exist_ok=True)
            if os.path.isdir(self.config.DEF_SPEAKER_DIR):
                def_voice_dir = self.config.DEF_SPEAKER_DIR
        _voice_dirs = [def_voice_dir] if def_voice_dir is not None else []
        if voice_dirs is not None:
            if isinstance(voice_dirs, str):
                voice_dirs = [voice_dirs]
            _voice_dirs = voice_dirs + _voice_dirs
        return _voice_dirs

    def synthesize(self, text, config, speaker_id="random", voice_dirs=None, **kwargs):
        speaker_id = "random" if speaker_id is None else speaker_id
        voice_dirs = self._set_voice_dirs(voice_dirs)
        history_prompt = load_voice(self, speaker_id, voice_dirs)
        outputs = self.generate_audio(text, history_prompt=history_prompt, **kwargs)
        return_dict = {
            "wav": outputs[0],
            "text_inputs": text,
        }

        return return_dict

    def eval_step(self):
        pass

    def forward(self):
        pass

    def inference(self):
        pass

    @staticmethod
    def init_from_config(config: "BarkConfig", **kwargs):
        return Bark(config)

    def load_checkpoint(self, config, checkpoint_dir, text_model_path=None, coarse_model_path=None, fine_model_path=None, hubert_model_path=None, hubert_tokenizer_path=None, eval=False, strict=True, **kwargs):
        text_model_path = text_model_path or os.path.join(checkpoint_dir, "text_2.pt")
        coarse_model_path = coarse_model_path or os.path.join(checkpoint_dir, "coarse_2.pt")
        fine_model_path = fine_model_path or os.path.join(checkpoint_dir, "fine_2.pt")
        hubert_model_path = hubert_model_path or os.path.join(checkpoint_dir, "hubert.pt")
        hubert_tokenizer_path = hubert_tokenizer_path or os.path.join(checkpoint_dir, "tokenizer.pth")

        self.config.LOCAL_MODEL_PATHS["text"] = text_model_path
        self.config.LOCAL_MODEL_PATHS["coarse"] = coarse_model_path
        self.config.LOCAL_MODEL_PATHS["fine"] = fine_model_path
        self.config.LOCAL_MODEL_PATHS["hubert"] = hubert_model_path
        self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"] = hubert_tokenizer_path

        self.load_bark_models()

        if eval:
            self.eval()
your_semantic_config = SemanticConfig(num_layers=6, embedding_size=512, vocab_size=30000)
# Assuming you have your BarkConfig instance named 'your_bark_config'

your_bark_config = BarkAudioConfig(num_chars=0, model_args={'your': 'model_args'})

your_coarse_config = CoarseConfig(num_layers=6, embedding_size=512, vocab_size=30000)

your_fine_config = FineConfig(num_layers=6, embedding_size=512, vocab_size=30000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create instances of your TTS and voice cloning models
tts_model = Bark(config=your_bark_config, tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased')).to(device)







from transformers import BertTokenizer
from dataclasses import dataclass
from coqpit import Coqpit
from TTS.tts.models.base_tts import BaseTTS
from typing import Optional
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from TTS.tts.models.bark import Bark
from encodec import EncodecModel
from TTS.tts.models.bark import BarkAudioConfig
from dataclasses import dataclass, field
import torch.nn.functional as F
from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
# Define your voice cloning model
class VoiceCloningDataset(Dataset):
    def __init__(self, text_file_path, audio_data_dir):
        self.samples = self.load_dataset(text_file_path, audio_data_dir)

    def load_dataset(self, text_file_path, audio_data_dir):
        samples = []
        with open(text_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    file_path, text = line.strip().split('|')
                    audio_path = os.path.join(audio_data_dir, file_path)
                    samples.append((audio_path, text))
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Error details: {e}")
        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, text = self.samples[index]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, text
        except Exception as e:
            print(f"Error loading audio file '{audio_path}': {e}")
            return torch.zeros(1), text  # Return a placeholder tensor if loading fails


def collate_fn(batch):
    # Separate text tokens and audio waveforms
    text_tokens, audio_waveforms = zip(*batch)

    # Load and process audio waveforms
    audio_waveforms = [torchaudio.load(waveform)[0] for waveform in audio_waveforms]

    # Pad audio waveforms to the length of the longest waveform in the batch
    max_len = max(waveform.size(1) for waveform in audio_waveforms)
    audio_waveforms = [F.pad(waveform, (0, max_len - waveform.size(1))) for waveform in audio_waveforms]
    audio_waveforms = torch.stack(audio_waveforms).squeeze(1)

    # Pad text tokens
    text_tokens = pad_sequence(text_tokens, batch_first=True)

    return text_tokens, audio_waveforms


class CombinedModel(nn.Module):
    def __init__(self, tts_hidden_size, voice_cloning_hidden_size, output_size):
        super(CombinedModel, self).__init__()
        config = BarkAudioConfig()
        print(config.__dict__)
        # Instantiate your TTS and voice cloning models
        self.tts_model = Bark(config=BarkAudioConfig()).to(device)
        
        self.voice_cloning_model = EncodecModel.encodec_model_24khz().to(device)

        # Linear layer for combining TTS and voice cloning outputs
        self.combine_layer = nn.Linear(tts_hidden_size + voice_cloning_hidden_size, output_size)

    def forward(self, text_tokens):
        # TTS model forward pass
        tts_outputs = self.tts_model(text_tokens)

        # Voice cloning model forward pass
        voice_cloning_outputs = self.voice_cloning_model(tts_outputs)

        # Combine TTS and voice cloning outputs
        combined_output = self.combine_layer(torch.cat((tts_outputs, voice_cloning_outputs), dim=1))

        return combined_output


# Adjust hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32
hidden_size = 256
output_size = 128

# Create the combined model
combined_model = CombinedModel(
    tts_hidden_size=hidden_size,
    voice_cloning_hidden_size=hidden_size,
    output_size=output_size
).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

# Load your dataset
audio_data_dir = "C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\wavs_harsha"
text_file_path = "C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\list_har - Copy.txt"
dataset = VoiceCloningDataset(text_file_path, audio_data_dir)
dataset = [sample for sample in dataset if sample is not None]
batch_size = 4
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Check for None values and filter them out
train_loader = DataLoader([x for x in dataset if x is not None], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Train the model
def train_combined_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (text_tokens, audio_waveform) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()

            try:
                # Move data to the device if you're using GPU
                print("File path:", text_tokens) 
                text_tokens, audio_waveform = text_tokens.to(device), audio_waveform.to(device)

                tts_outputs = model.tts_model(text_tokens)
                voice_cloning_outputs = model.voice_cloning_model(tts_outputs)

                # Define your loss based on the TTS and voice cloning outputs
                loss = criterion(voice_cloning_outputs, audio_waveform)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Error processing file: {text_tokens}")
                print(f"Error details: {e}")
                # Optionally, save problematic file paths to investigate further
                with open("problematic_files.txt", "a") as f:
                    f.write(f"Batch {batch_idx}: {audio_file_path}\n")
                    f.write(f"Text: {text_tokens}\n")
                    f.write(f"Audio Shape: {audio_waveform.shape}, Sample Rate: {sample_rate}\n")

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}')

# Train the combined model
train_combined_model(combined_model, train_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the trained combined model
torch.save(combined_model.state_dict(), 'combined_model.pth')
print("Trained combined model saved.")

# Inference and Save Modified Speech
input_text = "Example text for voice cloning."
# Assuming you have a BERT tokenizer instance
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(device)

modified_speech = combined_model(input_tokens)["modified_speech"]

# Save the modified speech
sf.write('modified_voice.wav', modified_speech.cpu().numpy(), 16000)
print("Modified voice saved.")
