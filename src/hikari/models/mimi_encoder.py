"""Mimi-based encoder for Hikari."""

import torch.nn as nn
from transformers import MimiModel
from transformers.modeling_utils import ModuleUtilsMixin


class WhisperMimiEncoder(nn.Module, ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.mimi_encoder = MimiModel.from_pretrained("kyutai/mimi", wo_decoder=True)
        self.adapter = nn.Linear(config.hidden_size, 1024, bias=False)
        self._init_weights(self.adapter)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs, **kwargs):
        outputs = self.mimi_encoder.encode(inputs)
        proj_embeddings = self.adapter(outputs.embeddings.transpose(1, 2))

        outputs.last_hidden_state = proj_embeddings

        return outputs
