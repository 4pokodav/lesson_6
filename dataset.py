from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        ids = tokenizer.encode(text).ids
        self.inputs = []

        for i in range(0, len(ids) - seq_len):
            x = ids[i:i + seq_len]
            self.inputs.append(torch.tensor(x, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        return {
            "input": x[:-1],
            "target": x[1:]
        }