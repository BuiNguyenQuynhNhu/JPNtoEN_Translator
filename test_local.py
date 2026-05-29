import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import logging

logging.basicConfig(level=logging.INFO)

config = AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_config(config)

input_ids = torch.randint(0, 100, (1, 10))
attention_mask = torch.ones(1, 10)
labels = torch.randint(0, 100, (1, 10))

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    output_hidden_states=True,
    return_dict=True
)

print(outputs.keys())
print("Has encoder_hidden_states:", hasattr(outputs, 'encoder_hidden_states'))
if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
    print("encoder_hidden_states type:", type(outputs.encoder_hidden_states))
    print("encoder_hidden_states len:", len(outputs.encoder_hidden_states))
    print("encoder_hidden_states[-1] shape:", outputs.encoder_hidden_states[-1].shape)

print("Has encoder_last_hidden_state:", hasattr(outputs, 'encoder_last_hidden_state'))
if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
    print("encoder_last_hidden_state shape:", outputs.encoder_last_hidden_state.shape)
