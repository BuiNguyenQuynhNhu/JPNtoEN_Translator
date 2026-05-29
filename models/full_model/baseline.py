"""
models/full_model/baseline.py

This module implements the Baseline NLLB/mBART wrapper for translation.
It exposes the encoder hidden states (Stage 5) to prepare for graph construction.

How to test:
    import torch
    from models.full_model.baseline import BaselineTranslator
    model = BaselineTranslator("facebook/nllb-200-distilled-600M")
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    labels = torch.randint(0, 1000, (2, 10))
    out = model(input_ids, attention_mask, labels)
    print(out.loss) # Scalar
    print(out.encoder_last_hidden_state.shape) # Expected: [2, 10, 1024]
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class BaselineTranslator(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Forward pass for training.
        
        Expected Tensor Shapes:
        - input_ids: [B, T]
        - attention_mask: [B, T]
        - labels: [B, T]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # We explicitly extract encoder hidden states for later graph usage (Stage 5)
        # encoder_hidden_states is a tuple of all layers, we take the last layer.
        # Ensure we have encoder hidden states
        if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
            encoder_last_hidden_state = outputs.encoder_hidden_states[-1] 
            # Expected shape: [B, T, D]
        else:
            # Fallback if model doesn't return tuple (though mBART/NLLB does when output_hidden_states=True)
            encoder_last_hidden_state = outputs.encoder_last_hidden_state

        outputs.encoder_last_hidden_state = encoder_last_hidden_state
        return outputs
        
    def generate(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass for inference/generation.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

# Future improvements:
# - Add LoRA support if requested.
# - The model can later be injected with graph modules.
