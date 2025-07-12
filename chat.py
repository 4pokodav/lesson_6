import torch
from generator_transformer import GeneratorTransformer
from tokenizers import Tokenizer


def sample_next_token(logits, temperature=1.0, top_k=10, used_tokens=None):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, top_k)

    if used_tokens is not None:
        for i, token_id in enumerate(top_k_indices[0]):
            if token_id.item() in used_tokens:
                top_k_logits[0][i] -= 1.0

    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
    next_token = top_k_indices.gather(1, torch.multinomial(probs, num_samples=1))
    return next_token.squeeze(1)

def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("homework_6/mistral_tokenizer.json")
    tokenizer.add_special_tokens(["<pad>", "<s>", "</s>"])

    model = GeneratorTransformer(
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=6,
        vocab_size=tokenizer.get_vocab_size(),
        pad_index=tokenizer.token_to_id("<pad>"),
        dropout=0.1,
        max_len=128,
        tokenizer=tokenizer,
        device=device
    ).to(device)

    checkpoint_path = "homework_6/checkpoints_lm/lm_epoch_2.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Введите начало текста (или 'quit' для выхода)")
    print("Введите '/beam' для включения beam search, '/sample' — для генерации с sampling (по умолчанию)")
    
    use_beam = False

    while True:
        user_input = input("Вы: ").strip()

        if user_input.lower() == 'quit':
            break
        if user_input.lower() == '/beam':
            use_beam = True
            print("[+] Beam search ВКЛЮЧЁН")
            continue
        if user_input.lower() == '/sample':
            use_beam = False
            print("[+] Beam search ВЫКЛЮЧЕН, используется sampling")
            continue

        # Токенизация
        input_ids = [tokenizer.token_to_id("<s>")] + tokenizer.encode(user_input).ids
        input_tensor = torch.tensor([input_ids], device=device)

        if use_beam:
            # Используем beam search
            output_ids = model.beam_search(input_tensor, beam_width=5)
        else:
            # Используем жадную генерацию с sampling
            generated = torch.full((1, model.max_len), tokenizer.token_to_id("<pad>"), dtype=torch.long, device=device)
            generated[:, :len(input_ids)] = input_tensor
            finished = torch.zeros(1, dtype=torch.bool, device=device)
            used_tokens = set(input_ids)

            for step in range(len(input_ids), model.max_len):
                if finished.all():
                    break
                logits = model.forward(generated[:, :step])
                next_token = sample_next_token(logits[:, -1, :], temperature=1.2, top_k=50, used_tokens=used_tokens)
                used_tokens.add(next_token.item())

                if next_token.item() in [tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("<s>")]:
                    continue

                generated[:, step] = next_token
                if next_token.item() == tokenizer.token_to_id("</s>"):
                    finished[:] = True

            output_ids = generated

        # Декодирование и вывод
        response = tokenizer.decode_batch(output_ids.tolist())[0]
        response = response.replace("<s>", "").replace("</s>", "").strip()
        if user_input in response:
            response = response.split(user_input, 1)[-1].strip()

        print(f"Модель: {response if response else '[пусто]'}")


if __name__ == "__main__":
    chat()