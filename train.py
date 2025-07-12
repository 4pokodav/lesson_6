import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from generator_transformer import GeneratorTransformer
from dataset import TextDataset
from torch import nn, optim
from tqdm import tqdm
import os

def main():
    # Настройки
    seq_len = 128
    batch_size = 64
    num_epochs = 2
    learning_rate = 1e-4
    save_path = "homework_6/checkpoints_lm"
    os.makedirs(save_path, exist_ok=True)

    # Загрузка токенизатора
    tokenizer = Tokenizer.from_file("homework_6/mistral_tokenizer.json")
    tokenizer.add_special_tokens(["<pad>", "<s>", "</s>"])

    # Загрузка текста
    with open("homework_6/data/book.txt", "r", encoding="utf-8") as f:
        text = f.read()

    dataset = TextDataset(text, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GeneratorTransformer(
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=6,
        vocab_size=tokenizer.get_vocab_size(),
        pad_index=tokenizer.token_to_id("<pad>"),
        dropout=0.1,
        max_len=seq_len,
        tokenizer=tokenizer,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress:
            input_ids = batch["input"].to(device)
            target_ids = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=total_loss / (progress.n + 1))

        torch.save(model.state_dict(), f"{save_path}/lm_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()