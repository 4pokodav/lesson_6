import torch
from typing import Optional
from layers import Decoder, DecoderLayer, MultiheadAttention, FeedForward, Embedding
from transformer import get_pad_mask, get_subsequent_mask
from tokenizers import Tokenizer
import torch.nn.functional as F


class GeneratorTransformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        vocab_size: int = 1000,
        pad_index: int = 0,
        dropout: float = 0.1,
        max_len: int = 64,
        tokenizer: Optional[Tokenizer] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.pad_index = pad_index
        self.tokenizer = tokenizer

        mha = MultiheadAttention(d_model, num_heads)
        ffn = FeedForward(d_model, d_ff)
        dec_layer = DecoderLayer(mha, mha, ffn, dropout)
        self.decoder = Decoder(dec_layer, num_layers)
        self.embedding = Embedding(d_model, vocab_size, pad_index)
        self.norm = torch.nn.LayerNorm(d_model)
        self.projection = torch.nn.Linear(d_model, vocab_size)

    def get_tgt_mask(self, x: torch.Tensor):
        pad_mask = get_pad_mask(x, self.pad_index).to(x.device)
        subsequent_mask = get_subsequent_mask(x).to(x.device)
        return pad_mask & subsequent_mask

    def forward(self, tgt_ids: torch.Tensor):
        tgt_mask = self.get_tgt_mask(tgt_ids)
        x = self.embedding(tgt_ids)
        x = self.decoder(x, x, None, tgt_mask)
        x = self.norm(x)
        return self.projection(x)

    def generate(self, prompt_ids: torch.Tensor):
        batch_size = prompt_ids.size(0)
        max_len = self.max_len
        pad_id = self.tokenizer.token_to_id("<pad>")
        start_id = self.tokenizer.token_to_id("<s>")
        end_id = self.tokenizer.token_to_id("</s>")

        generated = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=prompt_ids.device)
        generated[:, 0] = start_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)

        for step in range(1, max_len):
            if finished.all():
                break
            context_len = context_len = self.max_len
            output = self.forward(generated[:, max(0, step - context_len):step])
            next_token = output[:, -1, :].argmax(dim=-1)
            generated[:, step] = next_token
            finished |= (next_token == end_id)

        return generated
    
    def beam_search(self, prompt_ids: torch.Tensor, beam_width: int = 5, max_len: Optional[int] = None):
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        max_len = max_len or self.max_len

        pad_id = self.tokenizer.token_to_id("<pad>")
        start_id = self.tokenizer.token_to_id("<s>")
        end_id = self.tokenizer.token_to_id("</s>")

        # Инициализация: <s> в каждом луче
        sequences = torch.full((batch_size, beam_width, 1), start_id, dtype=torch.long, device=device)
        sequence_scores = torch.zeros(batch_size, beam_width, device=device)
        sequence_scores[:, 1:] = -float('inf')  # Только первый луч активен
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

        for step in range(1, max_len):
            seq_input = sequences.view(batch_size * beam_width, -1)
            logits = self.forward(seq_input)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            log_probs = log_probs.view(batch_size, beam_width, -1)  # (B, beam, vocab)

            length_penalty = ((5.0 + step) / 6.0) ** 1.0
            scores = sequence_scores.unsqueeze(2) + log_probs
            scores = scores / length_penalty  # применяем length penalty

            # topk по всем возможным beam * vocab
            scores = scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(scores, beam_width, dim=1)

            beam_indices = top_indices // log_probs.size(2)
            token_indices = top_indices % log_probs.size(2)

            new_sequences = []
            new_finished = []
            for b in range(batch_size):
                seqs = []
                fin_flags = []
                for i in range(beam_width):
                    seq = sequences[b, beam_indices[b, i]]
                    token = token_indices[b, i].unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=0)
                    seqs.append(new_seq.unsqueeze(0))
                    fin_flags.append(finished[b, beam_indices[b, i]] | (token == end_id))
                new_sequences.append(torch.cat(seqs, dim=0).unsqueeze(0))
                new_finished.append(torch.stack(fin_flags).unsqueeze(0))

            sequences = torch.cat(new_sequences, dim=0)
            finished = torch.cat(new_finished, dim=0)
            sequence_scores = top_scores * length_penalty  # восстанавливаем оригинальный score

            if finished.all():
                break

        best_seq = sequences[:, 0, :]
        return best_seq
