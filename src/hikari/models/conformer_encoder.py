import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder
from transformers.modeling_utils import ModuleUtilsMixin


class WhisperConformerEncoder(nn.Module, ModuleUtilsMixin):
    """A Conformer Encoder adapted to match Whisper's architecture and behavior.
    `ModuleUtilsMixin` is used to support gradient checkpointing
    """

    def __init__(self, config=None, input_channels=80, d_model=1024, num_layers=12):
        super().__init__()

        self.gradient_checkpointing = False

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.gelu = nn.GELU()

        if config is None:
            config = Wav2Vec2ConformerConfig(
                hidden_size=d_model,
                num_hidden_layers=12,
                num_attention_heads=16,
                intermediate_size=4096,
                hidden_act="swish",
                attention_dropout=0.1,
                position_embeddings_type="rotary",
                is_causal=True,
            )

        self.conformer_encoder = Wav2Vec2ConformerEncoder(config)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, mel_spectrogram, **kwargs):
        x = self.conv1(mel_spectrogram)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)

        x = x.transpose(1, 2)

        outputs = self.conformer_encoder(x)

        hidden_states = self.layer_norm(outputs.last_hidden_state)

        outputs.last_hidden_state = hidden_states

        return outputs
