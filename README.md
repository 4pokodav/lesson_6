# Домашнее задание: Генератор текста на базе Transformer

### 1. Архитектура модели

Создан класс `GeneratorTransformer`, реализующий декодерную часть Transformer. Он генерирует текст в авторегрессионном режиме, предсказывая следующий токен на основе предыдущих.

### 2. Данные

В качестве данных использован первый том книги "Война и мир" в английском переводе. Текст сохранён в `data/book.txt`.

### 3. Токенизация

Для токенизации использован готовый токенизатор `mistral_tokenizer.json`, содержащий необходимые специальные токены (`<pad>`, `<s>`, `</s>`). Токенизатор добавлен вручную и загружается через библиотеку `tokenizers`.

### 4. Обучение

**Параметры обучения:**
- `batch_size = 64`
- `max_length = 128`
- `learning_rate = 1e-4`
- `num_epochs = 2` (по причине большой длительности обучения)
- `dropout = 0.1`

Обучение проводилось с использованием функции потерь `CrossEntropyLoss` и оптимизатора `Adam`. Предсказываемая последовательность была сдвинута на один токен вперёд относительно входа, что соответствует задаче предсказания следующего токена.

### 5. Авторегрессивная генерация

Реализован метод `generate`, который строит продолжение входной последовательности по одному токену за шаг:

```python
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
```

### 6. Интерфейс для тестирования
Создан CLI-интерфейс `chat.py`, который позволяет пользователю вводить начало текста и получать продолжение от модели. В генерации используются temperature sampling и top-k фильтрация.

Пример взаимодействия с моделью:
!график

### 7. Результаты
!график

Во время валидации наблюдались низкие значения BLEU (менее 0.35). 

Возможные причины:
- Модель обучалась на маленьком объеме данных (одна книга),
- Архитектура не до конца оптимизирована для генерации длинных

Примеры проблемной генерации:
– Модель склонна к повторам одних и тех же слов или токенов (например, "benef benef benef..." или "The The The...").

### 8. Выводы
- Модель успешно обучается и может генерировать текст, хоть и плохо.
- Низкие значения BLEU показывают, что текущее обучение недостаточно для генерации осмысленного текста.