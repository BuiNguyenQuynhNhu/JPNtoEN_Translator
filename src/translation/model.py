import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerNMT(nn.Module):
    def __init__(self, vocab_size:int, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512, tie_embeddings=True):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(d_model, vocab_size)
        if tie_embeddings:
            self.generator.weight = self.tgt_tok_emb.weight

    def forward(self, src, tgt_input, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # src: (S,B), tgt_input: (T,B)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.tgt_tok_emb(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        memory = self.transformer.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        out = self.transformer.decoder(tgt_emb.transpose(0,1), memory,
                                       tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
        logits = self.generator(out.transpose(0,1))
        return logits  # (B, T, V)

def generate_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask
