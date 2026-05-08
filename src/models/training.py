import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from src.models.model import TransformerNMT, generate_mask
from src.translation.data.kftt import KFTTDataset, kftt_spm
import os
from tqdm.auto import tqdm
import time
import yaml

with open("configs/translation.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"])

# data config
data_dir = config["data"]["KFTT"]
max_len = config["data"]["MAX_LEN"]

# model config
sp_model = config["model"]["SP_MODEL"]
vocab_size = config["model"]["VOCAB_SIZE"]
batch_size = config["model"]["BATCH_SIZE"]
epochs = config["model"]["EPOCHS"]
num_worker = config["model"]["NUM_WORKERS"]
lr = float(config["model"]["LEARNING_RATE"])
ignore_index = config["model"]["IGNORE_INDEX"]

# datasets + loaders
train_ds = KFTTDataset(data_dir, split="train", sp_model_path=sp_model, train_sp_model=False, max_length=max_len, add_bos_eos=True, ignore_index=ignore_index)
dev_ds = KFTTDataset(data_dir, split="dev", sp_model_path=sp_model, train_sp_model=False, max_length=max_len, add_bos_eos=True, ignore_index=ignore_index)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=kftt_spm)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=kftt_spm)

# model
model = TransformerNMT(vocab_size=vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=max_len, tie_embeddings=True)
model.to(device)

# optimizer + loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = CrossEntropyLoss(ignore_index=ignore_index)

for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    processed = 0
    for batch in train_iter:
        src = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # chuẩn bị tgt_input như trước
        tgt_input = labels.clone()
        tgt_input[tgt_input == ignore_index] = 0
        tgt_input = torch.roll(tgt_input, shifts=1, dims=1)
        tgt_input[:, 0] = 2  # BOS id

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt_input == 0)
        tgt_mask = generate_mask(tgt_input.size(1)).to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_input,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       tgt_mask=tgt_mask,
                       memory_key_padding_mask=src_key_padding_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss
        processed += 1

        lr = optimizer.param_groups[0]['lr']
        avg_loss_so_far = epoch_loss / processed
        train_iter.set_postfix({"batch_loss": f"{batch_loss:.4f}", "avg_loss": f"{avg_loss_so_far:.4f}", "lr": f"{lr:.2e}"})
    elapsed = time.time() - start_time
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} train loss {avg_train_loss:.4f} time {elapsed:.1f}s")

    model.eval()
    dev_loss = 0.0
    with torch.no_grad():
        dev_iter = tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [dev]", leave=False)
        for batch in dev_iter:
            src = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tgt_input = labels.clone()
            tgt_input[tgt_input == ignore_index] = 0
            tgt_input = torch.roll(tgt_input, shifts=1, dims=1)
            tgt_input[:, 0] = 2
            src_key_padding_mask = (src == 0)
            tgt_key_padding_mask = (tgt_input == 0)
            tgt_mask = generate_mask(tgt_input.size(1)).to(device)
            logits = model(src, tgt_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            dev_loss += loss.item()
    print(f"Epoch {epoch} dev loss {dev_loss/len(dev_loader):.4f}")

    # save checkpoint
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, f"checkpoint_epoch{epoch}.pt")
