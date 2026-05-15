import torch
import math
from src.translation.data.kftt import KFTTDataset
from src.models.decoder import TransformerNMT, generate_mask
import sentencepiece as spm

def greedy_decode(model, sp, src_text, max_len=128, device='cpu'):
    model.eval()
    src_ids = sp.encode(src_text, out_type=int)
    src_ids = [2] + src_ids + [3]  # add bos/eos if model expects
    if len(src_ids) > max_len: src_ids = src_ids[:max_len]
    src_ids = src_ids + [0] * (max_len - len(src_ids))
    src = torch.LongTensor([src_ids]).to(device)
    src_key_padding_mask = (src == 0)
    memory = model.transformer.encoder(model.pos_enc(model.src_tok_emb(src) * math.sqrt(model.d_model)).transpose(0,1), src_key_padding_mask=src_key_padding_mask)
    ys = torch.LongTensor([[2]]).to(device)  # start with BOS
    for i in range(max_len-1):
        tgt_mask = generate_mask(ys.size(1)).to(device)
        out = model.transformer.decoder(model.pos_enc(model.tgt_tok_emb(ys) * math.sqrt(model.d_model)).transpose(0,1), memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0,1)[:, -1, :]  # (B, d_model)
        prob = model.generator(out)  # (B, V)
        next_token = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.LongTensor([[next_token]]).to(device)], dim=1)
        if next_token == 3:  # EOS
            break
    ids = ys.squeeze().tolist()
    # remove BOS/EOS
    if ids and ids[0] == 2: ids = ids[1:]
    if ids and ids[-1] == 3: ids = ids[:-1]
    return sp.decode(ids)

# usage
sp = spm.SentencePieceProcessor(); sp.load(sp_model)
model.eval()
print(greedy_decode(model, sp, "これはテストです。", max_len=128, device=device))
