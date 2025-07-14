## 1. Введение

### Описание задачи
В рамках данной работы проводится сравнение производительности моделей ResNet-18, конвертированных в различные форматы (TorchScript, ONNX, Torch-TensorRT), на различных размерах изображений. Цель — определить наиболее эффективный формат и конфигурацию для инференса.

### Методы оптимизации
Используются следующие методы оптимизации:
- Квантование до `fp16`
- Использование `torch-tensorrt` для ускорения инференса
- Перевод модели в формат ONNX для тестирования на CPU

### Ожидаемые результаты
- Повышение скорости инференса при использовании `torch-tensorrt`
- Снижение использования видеопамяти при использовании ONNX
- Явное превосходство `fp16` над `fp32` по скорости
- Визуальное и табличное сравнение эффективности всех подходов

## 2. Методология

### Экспериментальная установка
Тестирование проводилось на компьютере со следующей конфигурацией:
- Процессор: Intel Core i5-9600KF
- Видеокарта: MSI GeForce RTX 4060, 8 ГБ GDDR6

### Параметры тестирования
Для оценки производительности использовались различные размеры батча (32, 40, 48, 56, 64) и входные изображения размером 224×224, 256x256, 384x384, 512x512  пикселя. Тестирование проводилось как на CPU, так и на GPU.

### Методы измерения
Производительность измерялась по метрике FPS (кадров в секунду) и среднему времени инференса одного батча. Для замеров использовались встроенные средства cuda.Event (на GPU) и модуль time (на CPU).

3. Результаты

**Таблица бенчмарка с размером изображений 224x224:**

| function   | dataloader_type   | precision   | device   | shape             | time       | allocated_memory   | speedup   |
| torch      | real              | fp16        | cuda     | (32, 3, 224, 224) | 12.638 ms  | 345.6 MB           | 13.4x     |
| torch      | real              | fp16        | cuda     | (40, 3, 224, 224) | 24.041 ms  | 345.6 MB           | 7.1x      |
| torch      | real              | fp16        | cuda     | (48, 3, 224, 224) | 28.766 ms  | 345.6 MB           | 5.9x      |
| torch      | real              | fp16        | cuda     | (56, 3, 224, 224) | 34.362 ms  | 345.6 MB           | 4.9x      |
| torch      | real              | fp16        | cuda     | (64, 3, 224, 224) | 40.901 ms  | 345.6 MB           | 4.1x      |
| torch      | dummy             | fp16        | cuda     | (32, 3, 224, 224) | 20.161 ms  | 294.8 MB           | 8.4x      |
| torch      | dummy             | fp16        | cuda     | (40, 3, 224, 224) | 25.261 ms  | 294.8 MB           | 6.7x      |
| torch      | dummy             | fp16        | cuda     | (48, 3, 224, 224) | 30.090 ms  | 294.8 MB           | 5.6x      |
| torch      | dummy             | fp16        | cuda     | (56, 3, 224, 224) | 36.533 ms  | 294.8 MB           | 4.6x      |
| torch      | dummy             | fp16        | cuda     | (64, 3, 224, 224) | 41.639 ms  | 294.8 MB           | 4.1x      |
| onnx       | real              | fp16        | cuda     | (32, 3, 224, 224) | 33.093 ms  | 36.8 MB            | 5.1x      |
| onnx       | real              | fp16        | cuda     | (40, 3, 224, 224) | 40.713 ms  | 36.8 MB            | 4.2x      |
| onnx       | real              | fp16        | cuda     | (48, 3, 224, 224) | 49.601 ms  | 36.8 MB            | 3.4x      |
| onnx       | real              | fp16        | cuda     | (56, 3, 224, 224) | 57.918 ms  | 36.8 MB            | 2.9x      |
| onnx       | real              | fp16        | cuda     | (64, 3, 224, 224) | 66.838 ms  | 36.8 MB            | 2.5x      |
| onnx       | dummy             | fp16        | cuda     | (32, 3, 224, 224) | 32.644 ms  | 36.8 MB            | 5.2x      |
| onnx       | dummy             | fp16        | cuda     | (40, 3, 224, 224) | 39.868 ms  | 36.8 MB            | 4.3x      |
| onnx       | dummy             | fp16        | cuda     | (48, 3, 224, 224) | 48.503 ms  | 36.8 MB            | 3.5x      |
| onnx       | dummy             | fp16        | cuda     | (56, 3, 224, 224) | 56.364 ms  | 36.8 MB            | 3.0x      |
| onnx       | dummy             | fp16        | cuda     | (64, 3, 224, 224) | 54.467 ms  | 36.8 MB            | 3.1x      |
| torch_trt  | real              | fp16        | cuda     | (32, 3, 224, 224) | 13.760 ms  | 191.5 MB           | 12.3x     |
| torch_trt  | real              | fp16        | cuda     | (40, 3, 224, 224) | 15.989 ms  | 191.5 MB           | 10.6x     |
| torch_trt  | real              | fp16        | cuda     | (48, 3, 224, 224) | 18.336 ms  | 191.5 MB           | 9.2x      |
| torch_trt  | real              | fp16        | cuda     | (56, 3, 224, 224) | 20.553 ms  | 191.5 MB           | 8.2x      |
| torch_trt  | real              | fp16        | cuda     | (64, 3, 224, 224) | 22.787 ms  | 191.5 MB           | 7.4x      |
| torch_trt  | dummy             | fp16        | cuda     | (32, 3, 224, 224) | 13.746 ms  | 189.8 MB           | 12.3x     |
| torch_trt  | dummy             | fp16        | cuda     | (40, 3, 224, 224) | 15.989 ms  | 189.8 MB           | 10.6x     |
| torch_trt  | dummy             | fp16        | cuda     | (48, 3, 224, 224) | 18.236 ms  | 189.8 MB           | 9.3x      |
| torch_trt  | dummy             | fp16        | cuda     | (56, 3, 224, 224) | 20.435 ms  | 189.8 MB           | 8.3x      |
| torch_trt  | dummy             | fp16        | cuda     | (64, 3, 224, 224) | 22.678 ms  | 189.8 MB           | 7.5x      |
| torch      | real              | fp32        | cuda     | (32, 3, 224, 224) | 27.524 ms  | 465.5 MB           | 6.2x      |
| torch      | real              | fp32        | cuda     | (40, 3, 224, 224) | 33.347 ms  | 465.5 MB           | 5.1x      |
| torch      | real              | fp32        | cuda     | (48, 3, 224, 224) | 39.712 ms  | 465.5 MB           | 4.3x      |
| torch      | real              | fp32        | cuda     | (56, 3, 224, 224) | 45.531 ms  | 465.5 MB           | 3.7x      |
| torch      | real              | fp32        | cuda     | (64, 3, 224, 224) | 52.468 ms  | 465.5 MB           | 3.2x      |
| torch      | dummy             | fp32        | cuda     | (32, 3, 224, 224) | 27.217 ms  | 465.5 MB           | 6.2x      |
| torch      | dummy             | fp32        | cuda     | (40, 3, 224, 224) | 20.664 ms  | 465.5 MB           | 8.2x      |
| torch      | dummy             | fp32        | cuda     | (48, 3, 224, 224) | 24.960 ms  | 465.5 MB           | 6.8x      |
| torch      | dummy             | fp32        | cuda     | (56, 3, 224, 224) | 29.057 ms  | 465.5 MB           | 5.8x      |
| torch      | dummy             | fp32        | cuda     | (64, 3, 224, 224) | 33.890 ms  | 465.5 MB           | 5.0x      |
| onnx       | real              | fp32        | cuda     | (32, 3, 224, 224) | 15.971 ms  | 36.8 MB            | 10.6x     |
| onnx       | real              | fp32        | cuda     | (40, 3, 224, 224) | 20.025 ms  | 36.8 MB            | 8.5x      |
| onnx       | real              | fp32        | cuda     | (48, 3, 224, 224) | 24.288 ms  | 36.8 MB            | 7.0x      |
| onnx       | real              | fp32        | cuda     | (56, 3, 224, 224) | 28.186 ms  | 36.8 MB            | 6.0x      |
| onnx       | real              | fp32        | cuda     | (64, 3, 224, 224) | 33.029 ms  | 36.8 MB            | 5.1x      |
| onnx       | real              | fp32        | cpu      | (32, 3, 224, 224) | 169.514 ms | -338.8 MB          | 1.0x      |
| onnx       | real              | fp32        | cpu      | (40, 3, 224, 224) | 213.151 ms | -338.8 MB          | 0.8x      |
| onnx       | real              | fp32        | cpu      | (48, 3, 224, 224) | 254.724 ms | -338.8 MB          | 0.7x      |
| onnx       | real              | fp32        | cpu      | (56, 3, 224, 224) | 295.240 ms | -338.8 MB          | 0.6x      |
| onnx       | real              | fp32        | cpu      | (64, 3, 224, 224) | 334.080 ms | -338.8 MB          | 0.5x      |
| onnx       | dummy             | fp32        | cuda     | (32, 3, 224, 224) | 15.969 ms  | 36.8 MB            | 10.6x     |
| onnx       | dummy             | fp32        | cuda     | (40, 3, 224, 224) | 20.022 ms  | 36.8 MB            | 8.5x      |
| onnx       | dummy             | fp32        | cuda     | (48, 3, 224, 224) | 24.267 ms  | 36.8 MB            | 7.0x      |
| onnx       | dummy             | fp32        | cuda     | (56, 3, 224, 224) | 28.183 ms  | 36.8 MB            | 6.0x      |
| onnx       | dummy             | fp32        | cuda     | (64, 3, 224, 224) | 33.041 ms  | 36.8 MB            | 5.1x      |
| onnx       | dummy             | fp32        | cpu      | (32, 3, 224, 224) | 167.353 ms | -16.1 MB           | 1.0x      |
| onnx       | dummy             | fp32        | cpu      | (40, 3, 224, 224) | 207.623 ms | -16.1 MB           | 0.8x      |
| onnx       | dummy             | fp32        | cpu      | (48, 3, 224, 224) | 249.858 ms | -16.1 MB           | 0.7x      |
| onnx       | dummy             | fp32        | cpu      | (56, 3, 224, 224) | 287.951 ms | -16.1 MB           | 0.6x      |
| onnx       | dummy             | fp32        | cpu      | (64, 3, 224, 224) | 329.044 ms | -16.1 MB           | 0.5x      |
| torch_trt  | real              | fp32        | cuda     | (32, 3, 224, 224) | 13.512 ms  | 189.8 MB           | 12.5x     |
| torch_trt  | real              | fp32        | cuda     | (40, 3, 224, 224) | 16.642 ms  | 189.8 MB           | 10.2x     |
| torch_trt  | real              | fp32        | cuda     | (48, 3, 224, 224) | 19.729 ms  | 189.8 MB           | 8.6x      |
| torch_trt  | real              | fp32        | cuda     | (56, 3, 224, 224) | 23.029 ms  | 189.8 MB           | 7.4x      |
| torch_trt  | real              | fp32        | cuda     | (64, 3, 224, 224) | 26.405 ms  | 189.8 MB           | 6.4x      |
| torch_trt  | dummy             | fp32        | cuda     | (32, 3, 224, 224) | 13.466 ms  | 189.8 MB           | 12.6x     |
| torch_trt  | dummy             | fp32        | cuda     | (40, 3, 224, 224) | 16.629 ms  | 189.8 MB           | 10.2x     |
| torch_trt  | dummy             | fp32        | cuda     | (48, 3, 224, 224) | 19.689 ms  | 189.8 MB           | 8.6x      |
| torch_trt  | dummy             | fp32        | cuda     | (56, 3, 224, 224) | 22.978 ms  | 189.8 MB           | 7.4x      |
| torch_trt  | dummy             | fp32        | cuda     | (64, 3, 224, 224) | 26.379 ms  | 189.8 MB           | 6.4x      |

**Таблица бенчмарка с размером изображений 256x256:**

| function | dataloader_type | precision | device |      shape        | time      | allocated_memory | speedup |
| torch    | real            | fp16      | cuda   | (32, 3, 256, 526) | 21.214 ms | 395.1 MB         | 11.5x   |
| torch    | real            | fp16      | cuda   | (40, 3, 256, 526) | 32.055 ms | 387.7 MB         | 5.3x    |
| torch    | real            | fp16      | cuda   | (48, 3, 256, 256) | 36.490 ms | 390.4 MB         | 4.2x    |
| torch    | real            | fp16      | cuda   | (56, 3, 256, 256) | 41.480 ms | 376.4 MB         | 3.8x    |
| torch    | real            | fp16      | cuda   | (64, 3, 256, 256) | 49.130 ms | 381.3 MB         | 2.2x    |
| torch    | dummy           | fp16      | cuda   | (32, 3, 256, 256) | 27.349 ms | 327.2 MB         | 6.7x    |
| torch    | dummy           | fp16      | cuda   | (40, 3, 256, 256) | 34.720 ms | 330.7 MB         | 4.7x    |
| torch    | dummy           | fp16      | cuda   | (48, 3, 256, 256) | 39.908 ms | 327.2 MB         | 4.5x    |
| torch    | dummy           | fp16      | cuda   | (56, 3, 256, 256) | 43.450 ms | 331.2 MB         | 2.7x    |
| torch    | dummy           | fp16      | cuda   | (64, 3, 256, 256) | 50.598 ms | 333.1 MB         | 2.9x    |
| onnx     | real            | fp16      | cuda   | (32, 3, 256, 256) | 40.737 ms | 68.1 MB          | 3.5x    |
| onnx     | real            | fp16      | cuda   | (40, 3, 256, 256) | 48.553 ms | 80.6 MB          | 3.1x    |
| onnx     | real            | fp16      | cuda   | (48, 3, 256, 256) | 59.229 ms | 78.1 MB          | 1.6x    |
| onnx     | real            | fp16      | cuda   | (56, 3, 256, 256) | 63.273 ms | 72.1 MB          | 1.1x    |
| onnx     | real            | fp16      | cuda   | (64, 3, 256, 256) | 72.274 ms | 77.3 MB          | 0.9x    |
| onnx     | dummy           | fp16      | cuda   | (32, 3, 256, 256) | 37.745 ms | 68.7 MB          | 3.8x    |
| onnx     | dummy           | fp16      | cuda   | (40, 3, 256, 256) | 49.031 ms | 78.3 MB          | 3.2x    |
| onnx     | dummy           | fp16      | cuda   | (48, 3, 256, 256) | 57.394 ms | 85.4 MB          | 1.8x    |
| onnx     | dummy           | fp16      | cuda   | (56, 3, 256, 256) | 65.714 ms | 73.2 MB          | 1.5x    |
| onnx     | dummy           | fp16      | cuda   | (64, 3, 256, 256) | 64.360 ms | 80.1 MB          | 1.4x    |
| torch_trt| real            | fp16      | cuda   | (32, 3, 256, 256) | 22.756 ms | 224.1 MB         | 10.4x   |
| torch_trt| real            | fp16      | cuda   | (40, 3, 256, 256) | 23.296 ms | 235.8 MB         | 8.6x    |
| torch_trt| real            | fp16      | cuda   | (48, 3, 256, 256) | 27.239 ms | 227.3 MB         | 7.3x    |
| torch_trt| real            | fp16      | cuda   | (56, 3, 256, 256) | 26.144 ms | 225.2 MB         | 7.2x    |
| torch_trt| real            | fp16      | cuda   | (64, 3, 256, 256) | 30.987 ms | 233.2 MB         | 6.0x    |
| torch_trt| dummy           | fp16      | cuda   | (32, 3, 256, 256) | 19.463 ms | 220.2 MB         | 10.6x   |
| torch_trt| dummy           | fp16      | cuda   | (40, 3, 256, 256) | 25.712 ms | 236.4 MB         | 9.4x    |
| torch_trt| dummy           | fp16      | cuda   | (48, 3, 256, 256) | 25.845 ms | 219.9 MB         | 7.8x    |
| torch_trt| dummy           | fp16      | cuda   | (56, 3, 256, 256) | 27.508 ms | 233.4 MB         | 7.2x    |
| torch_trt| dummy           | fp16      | cuda   | (64, 3, 256, 256) | 29.001 ms | 225.2 MB         | 6.3x    |
| torch    | real            | fp32      | cuda   | (32, 3, 256, 256) | 36.395 ms | 510.2 MB         | 5.2x    |
| torch    | real            | fp32      | cuda   | (40, 3, 256, 256) | 40.628 ms | 514.7 MB         | 3.3x    |
| torch    | real            | fp32      | cuda   | (48, 3, 256, 256) | 47.554 ms | 500.5 MB         | 3.1x    |
| torch    | real            | fp32      | cuda   | (56, 3, 256, 256) | 50.625 ms | 507.0 MB         | 2.4x    |
| torch    | real            | fp32      | cuda   | (64, 3, 256, 256) | 60.556 ms | 507.3 MB         | 1.3x    |
| torch    | dummy           | fp32      | cuda   | (32, 3, 256, 256) | 35.277 ms | 506.9 MB         | 4.5x    |
| torch    | dummy           | fp32      | cuda   | (40, 3, 256, 256) | 28.749 ms | 500.0 MB         | 7.2x    |
| torch    | dummy           | fp32      | cuda   | (48, 3, 256, 256) | 34.679 ms | 514.6 MB         | 5.6x    |
| torch    | dummy           | fp32      | cuda   | (56, 3, 256, 256) | 37.466 ms | 504.4 MB         | 4.2x    |
| torch    | dummy           | fp32      | cuda   | (64, 3, 256, 256) | 40.688 ms | 512.4 MB         | 3.4x    |
| onnx     | real            | fp32      | cuda   | (32, 3, 256, 256) | 23.156 ms | 80.8 MB          | 9.4x    |
| onnx     | real            | fp32      | cuda   | (40, 3, 256, 256) | 28.513 ms | 72.7 MB          | 6.6x    |
| onnx     | real            | fp32      | cuda   | (48, 3, 256, 256) | 29.589 ms | 83.1 MB          | 5.4x    |
| onnx     | real            | fp32      | cuda   | (56, 3, 256, 256) | 36.520 ms | 74.7 MB          | 4.5x    |
| onnx     | real            | fp32      | cuda   | (64, 3, 256, 256) | 41.382 ms | 84.4 MB          | 3.5x    |
| onnx     | real            | fp32      | cpu    | (32, 3, 256, 256) | 175.566 ms | 380.4 MB        | -0.7x   |
| onnx     | real            | fp32      | cpu    | (40, 3, 256, 256) | 218.796 ms | 386.4 MB        | -0.5x   |
| onnx     | real            | fp32      | cpu    | (48, 3, 256, 256) | 261.301 ms | 382.7 MB        | -0.7x   |
| onnx     | real            | fp32      | cpu    | (56, 3, 256, 256) | 302.059 ms | 383.3 MB        | -0.6x   |
| onnx     | real            | fp32      | cpu    | (64, 3, 256, 256) | 341.931 ms | 378.8 MB        | -0.7x   |
| onnx     | dummy           | fp32      | cuda   | (32, 3, 256, 256) | 23.162 ms | 85.9 MB          | 8.7x    |
| onnx     | dummy           | fp32      | cuda   | (40, 3, 256, 256) | 29.964 ms | 79.7 MB          | 6.8x    |
| onnx     | dummy           | fp32      | cuda   | (48, 3, 256, 256) | 29.777 ms | 75.3 MB          | 5.5x    |
| onnx     | dummy           | fp32      | cuda   | (56, 3, 256, 256) | 34.227 ms | 78.9 MB          | 4.8x    |
| onnx     | dummy           | fp32      | cuda   | (64, 3, 256, 256) | 38.848 ms | 67.2 MB          | 3.8x    |
| onnx     | dummy           | fp32      | cpu    | (32, 3, 256, 256) | 175.619 ms | 52.1 MB         | -0.1x   |
| onnx     | dummy           | fp32      | cpu    | (40, 3, 256, 256) | 213.889 ms | 59.3 MB         | -0.6x   |
| onnx     | dummy           | fp32      | cpu    | (48, 3, 256, 256) | 257.190 ms | 51.9 MB         | -0.6x   |
| onnx     | dummy           | fp32      | cpu    | (56, 3, 256, 256) | 294.173 ms | 58.5 MB         | -1.1x   |
| onnx     | dummy           | fp32      | cpu    | (64, 3, 256, 256) | 334.839 ms | 54.7 MB         | -0.9x   |
| torch_trt| real            | fp32      | cuda   | (32, 3, 256, 256) | 19.064 ms | 222.5 MB         | 11.3x   |
| torch_trt| real            | fp32      | cuda   | (40, 3, 256, 256) | 24.924 ms | 225.8 MB         | 9.2x    |
| torch_trt| real            | fp32      | cuda   | (48, 3, 256, 256) | 25.420 ms | 231.2 MB         | 7.5x    |
| torch_trt| real            | fp32      | cuda   | (56, 3, 256, 256) | 29.012 ms | 231.6 MB         | 5.7x    |
| torch_trt| real            | fp32      | cuda   | (64, 3, 256, 256) | 33.249 ms | 231.3 MB         | 4.9x    |
| torch_trt| dummy           | fp32      | cuda   | (32, 3, 256, 256) | 22.571 ms | 232.9 MB         | 11.1x   |
| torch_trt| dummy           | fp32      | cuda   | (40, 3, 256, 256) | 22.115 ms | 232.8 MB         | 8.3x    |
| torch_trt| dummy           | fp32      | cuda   | (48, 3, 256, 256) | 28.879 ms | 228.4 MB         | 6.6x    |
| torch_trt| dummy           | fp32      | cuda   | (56, 3, 256, 256) | 28.458 ms | 237.7 MB         | 6.2x    |
| torch_trt| dummy           | fp32      | cuda   | (64, 3, 256, 256) | 36.261 ms | 227.2 MB         | 4.7x    |

**Таблица бенчмарка с размером изображений 384x384:**

| function | dataloader_type | precision | device |       shape       | time      | allocated_memory | speedup |
| torch    | real            | fp16      | cuda   | (32, 3, 384, 384) | 28.958 ms | 444.6 MB         | 9.6x    | 
| torch    | real            | fp16      | cuda   | (40, 3, 384, 384) | 40.631 ms | 427.1 MB         | 3.9x    | 
| torch    | real            | fp16      | cuda   | (48, 3, 384, 384) | 44.504 ms | 439.9 MB         | 2.8x    | 
| torch    | real            | fp16      | cuda   | (56, 3, 384, 384) | 49.204 ms | 418.5 MB         | 1.9x    | 
| torch    | real            | fp16      | cuda   | (64, 3, 384, 384) | 56.248 ms | 426.1 MB         | 0.4x    | 
| torch    | dummy           | fp16      | cuda   | (32, 3, 384, 384) | 35.578 ms | 358.0 MB         | 5.0x    | 
| torch    | dummy           | fp16      | cuda   | (40, 3, 384, 384) | 41.908 ms | 366.4 MB         | 3.6x    | 
| torch    | dummy           | fp16      | cuda   | (48, 3, 384, 384) | 49.367 ms | 359.6 MB         | 2.6x    | 
| torch    | dummy           | fp16      | cuda   | (56, 3, 384, 384) | 53.268 ms | 367.1 MB         | 1.0x    | 
| torch    | dummy           | fp16      | cuda   | (64, 3, 384, 384) | 57.515 ms | 365.5 MB         | 0.9x    | 
| onnx     | real            | fp16      | cuda   | (32, 3, 384, 384) | 49.696 ms | 104.5 MB         | 2.4x    | 
| onnx     | real            | fp16      | cuda   | (40, 3, 384, 384) | 56.197 ms | 118.9 MB         | 1.2x    | 
| onnx     | real            | fp16      | cuda   | (48, 3, 384, 384) | 67.069 ms | 109.4 MB         | 0.4x    | 
| onnx     | real            | fp16      | cuda   | (56, 3, 384, 384) | 72.901 ms | 115.9 MB         | -0.5x   | 
| onnx     | real            | fp16      | cuda   | (64, 3, 384, 384) | 77.629 ms | 118.6 MB         | -0.2x   | 
| onnx     | dummy           | fp16      | cuda   | (32, 3, 384, 384) | 43.181 ms | 104.0 MB         | 2.0x    | 
| onnx     | dummy           | fp16      | cuda   | (40, 3, 384, 384) | 54.132 ms | 118.8 MB         | 1.4x    | 
| onnx     | dummy           | fp16      | cuda   | (48, 3, 384, 384) | 66.557 ms | 117.3 MB         | 0.2x    | 
| onnx     | dummy           | fp16      | cuda   | (56, 3, 384, 384) | 74.605 ms | 114.7 MB         | 0.1x    | 
| onnx     | dummy           | fp16      | cuda   | (64, 3, 384, 384) | 73.710 ms | 128.7 MB         | 0.3x    | 
| torch_trt| real            | fp16      | cuda   | (32, 3, 384, 384) | 32.649 ms | 260.5 MB         | 8.7x    |
| torch_trt| real            | fp16      | cuda   | (40, 3, 384, 384) | 32.292 ms | 279.1 MB         | 7.1x    |
| torch_trt| real            | fp16      | cuda   | (48, 3, 384, 384) | 34.546 ms | 259.9 MB         | 5.6x    |
| torch_trt| real            | fp16      | cuda   | (56, 3, 384, 384) | 35.047 ms | 269.5 MB         | 5.3x    |
| torch_trt| real            | fp16      | cuda   | (64, 3, 384, 384) | 36.578 ms | 269.0 MB         | 4.0x    |
| torch_trt| dummy           | fp16      | cuda   | (32, 3, 384, 384) | 27.663 ms | 253.9 MB         | 8.7x    |
| torch_trt| dummy           | fp16      | cuda   | (40, 3, 384, 384) | 31.429 ms | 278.1 MB         | 8.4x    |
| torch_trt| dummy           | fp16      | cuda   | (48, 3, 384, 384) | 35.568 ms | 250.3 MB         | 6.4x    |
| torch_trt| dummy           | fp16      | cuda   | (56, 3, 384, 384) | 35.117 ms | 280.0 MB         | 5.5x    |
| torch_trt| dummy           | fp16      | cuda   | (64, 3, 384, 384) | 36.074 ms | 255.3 MB         | 5.1x    |
| torch    | real            | fp32      | cuda   | (32, 3, 384, 384) | 42.718 ms | 553.8 MB         | 3.7x    |
| torch    | real            | fp32      | cuda   | (40, 3, 384, 384) | 49.499 ms | 550.1 MB         | 2.2x    |
| torch    | real            | fp32      | cuda   | (48, 3, 384, 384) | 54.835 ms | 545.2 MB         | 1.9x    |
| torch    | real            | fp32      | cuda   | (56, 3, 384, 384) | 58.467 ms | 556.2 MB         | 1.4x    |
| torch    | real            | fp32      | cuda   | (64, 3, 384, 384) | 65.650 ms | 542.3 MB         | -0.5x   |
| torch    | dummy           | fp32      | cuda   | (32, 3, 384, 384) | 43.365 ms | 548.4 MB         | 3.3x    |
| torch    | dummy           | fp32      | cuda   | (40, 3, 384, 384) | 36.809 ms | 541.8 MB         | 5.9x    |
| torch    | dummy           | fp32      | cuda   | (48, 3, 384, 384) | 42.764 ms | 556.0 MB         | 3.7x    |
| torch    | dummy           | fp32      | cuda   | (56, 3, 384, 384) | 47.185 ms | 538.9 MB         | 2.5x    |
| torch    | dummy           | fp32      | cuda   | (64, 3, 384, 384) | 49.097 ms | 561.5 MB         | 2.4x    |
| onnx     | real            | fp32      | cuda   | (32, 3, 384, 384) | 29.954 ms | 119.7 MB         | 8.2x    |
| onnx     | real            | fp32      | cuda   | (40, 3, 384, 384) | 35.698 ms | 119.6 MB         | 5.0x    |
| onnx     | real            | fp32      | cuda   | (48, 3, 384, 384) | 38.077 ms | 127.1 MB         | 3.8x    |
| onnx     | real            | fp32      | cuda   | (56, 3, 384, 384) | 41.821 ms | 110.6 MB         | 3.3x    |
| onnx     | real            | fp32      | cuda   | (64, 3, 384, 384) | 49.716 ms | 130.7 MB         | 1.6x    |
| onnx     | real            | fp32      | cpu    | (32, 3, 384, 384) | 183.919 ms | 418.3 MB        | -0.9x   |
| onnx     | real            | fp32      | cpu    | (40, 3, 384, 384) | 224.848 ms | 434.0 MB        | -1.0x   |
| onnx     | real            | fp32      | cpu    | (48, 3, 384, 384) | 266.946 ms | 424.3 MB        | -0.9x   |
| onnx     | real            | fp32      | cpu    | (56, 3, 384, 384) | 308.636 ms | 430.9 MB        | -1.1x   |
| onnx     | real            | fp32      | cpu    | (64, 3, 384, 384) | 348.750 ms | 422.7 MB        | -0.6x   |
| onnx     | dummy           | fp32      | cuda   | (32, 3, 384, 384) | 31.013 ms | 130.4 MB         | 7.3x    |
| onnx     | dummy           | fp32      | cuda   | (40, 3, 384, 384) | 37.157 ms | 119.7 MB         | 5.6x    |
| onnx     | dummy           | fp32      | cuda   | (48, 3, 384, 384) | 39.719 ms | 124.4 MB         | 4.3x    |
| onnx     | dummy           | fp32      | cuda   | (56, 3, 384, 384) | 39.737 ms | 121.8 MB         | 2.9x    |
| onnx     | dummy           | fp32      | cuda   | (64, 3, 384, 384) | 44.892 ms | 105.7 MB         | 2.1x    |
| onnx     | dummy           | fp32      | cpu    | (32, 3, 384, 384) | 181.426 ms | 94.2 MB         | -1.4x   |
| onnx     | dummy           | fp32      | cpu    | (40, 3, 384, 384) | 222.155 ms | 89.7 MB         | -0.6x   |
| onnx     | dummy           | fp32      | cpu    | (48, 3, 384, 384) | 263.456 ms | 87.9 MB         | -0.7x   |
| onnx     | dummy           | fp32      | cpu    | (56, 3, 384, 384) | 301.505 ms | 101.7 MB        | 0.0x    |
| onnx     | dummy           | fp32      | cpu    | (64, 3, 384, 384) | 341.061 ms | 90.5 MB         | -0.5x   |
| torch_trt| real            | fp32      | cuda   | (32, 3, 384, 384) | 24.859 ms | 264.9 MB         | 10.0x   |
| torch_trt| real            | fp32      | cuda   | (40, 3, 384, 384) | 30.476 ms | 264.4 MB         | 7.5x    |
| torch_trt| real            | fp32      | cuda   | (48, 3, 384, 384) | 33.702 ms | 263.9 MB         | 6.1x    |
| torch_trt| real            | fp32      | cuda   | (56, 3, 384, 384) | 34.703 ms | 267.6 MB         | 4.5x    |
| torch_trt| real            | fp32      | cuda   | (64, 3, 384, 384) | 39.232 ms | 272.7 MB         | 3.9x    |
| torch_trt| dummy           | fp32      | cuda   | (32, 3, 384, 384) | 29.415 ms | 274.7 MB         | 10.0x   |
| torch_trt| dummy           | fp32      | cuda   | (40, 3, 384, 384) | 31.220 ms | 274.3 MB         | 6.6x    |
| torch_trt| dummy           | fp32      | cuda   | (48, 3, 384, 384) | 34.365 ms | 271.5 MB         | 5.1x    |
| torch_trt| dummy           | fp32      | cuda   | (56, 3, 384, 384) | 37.648 ms | 280.7 MB         | 4.7x    |
| torch_trt| dummy           | fp32      | cuda   | (64, 3, 384, 384) | 41.741 ms | 265.8 MB         | 2.8x    |

**Таблица бенчмарка с размером изображений 512x512:**

| function | dataloader_type | precision | device | shape             | time      | allocated_memory | speedup |
| torch    | real            | fp16      | cuda   | (32, 3, 512, 512) | 36.702 ms | 494.1 MB         | 7.7x    | 
| torch    | real            | fp16      | cuda   | (40, 3, 512, 512) | 49.207 ms | 466.5 MB         | 2.5x    | 
| torch    | real            | fp16      | cuda   | (48, 3, 512, 512) | 52.518 ms | 489.4 MB         | 1.4x    | 
| torch    | real            | fp16      | cuda   | (56, 3, 512, 512) | 56.928 ms | 460.6 MB         | 0.0x    | 
| torch    | real            | fp16      | cuda   | (64, 3, 512, 512) | 63.366 ms | 470.9 MB         | -1.4x   | 
| torch    | dummy           | fp16      | cuda   | (32, 3, 512, 512) | 43.807 ms | 388.8 MB         | 3.3x    | 
| torch    | dummy           | fp16      | cuda   | (40, 3, 512, 512) | 49.096 ms | 402.1 MB         | 2.5x    | 
| torch    | dummy           | fp16      | cuda   | (48, 3, 512, 512) | 58.826 ms | 392.0 MB         | 0.7x    | 
| torch    | dummy           | fp16      | cuda   | (56, 3, 512, 512) | 63.086 ms | 403.0 MB         | -0.7x   | 
| torch    | dummy           | fp16      | cuda   | (64, 3, 512, 512) | 64.432 ms | 397.9 MB         | -1.1x   | 
| onnx     | real            | fp16      | cuda   | (32, 3, 512, 512) | 58.655 ms | 140.9 MB         | 1.3x    | 
| onnx     | real            | fp16      | cuda   | (40, 3, 512, 512) | 63.841 ms | 157.2 MB         | -0.7x   | 
| onnx     | real            | fp16      | cuda   | (48, 3, 512, 512) | 74.909 ms | 140.7 MB         | -0.8x   | 
| onnx     | real            | fp16      | cuda   | (56, 3, 512, 512) | 82.529 ms | 159.7 MB         | -1.1x   | 
| onnx     | real            | fp16      | cuda   | (64, 3, 512, 512) | 82.984 ms | 159.9 MB         | -0.9x   | 
| onnx     | dummy           | fp16      | cuda   | (32, 3, 512, 512) | 48.617 ms | 139.3 MB         | 0.2x    | 
| onnx     | dummy           | fp16      | cuda   | (40, 3, 512, 512) | 59.233 ms | 159.3 MB         | -0.4x   | 
| onnx     | dummy           | fp16      | cuda   | (48, 3, 512, 512) | 75.720 ms | 149.2 MB         | -1.4x   | 
| onnx     | dummy           | fp16      | cuda   | (56, 3, 512, 512) | 83.496 ms | 156.2 MB         | -1.3x   | 
| onnx     | dummy           | fp16      | cuda   | (64, 3, 512, 512) | 83.060 ms | 177.3 MB         | -0.8x   | 
| torch_trt| real            | fp16      | cuda   | (32, 3, 512, 512) | 42.542 ms | 296.9 MB         | 7.0x    |
| torch_trt| real            | fp16      | cuda   | (40, 3, 512, 512) | 41.288 ms | 322.4 MB         | 5.6x    |
| torch_trt| real            | fp16      | cuda   | (48, 3, 512, 512) | 41.853 ms | 292.5 MB         | 3.9x    |
| torch_trt| real            | fp16      | cuda   | (56, 3, 512, 512) | 43.950 ms | 313.8 MB         | 3.4x    |
| torch_trt| real            | fp16      | cuda   | (64, 3, 512, 512) | 42.169 ms | 304.8 MB         | 2.0x    |
| torch_trt| dummy           | fp16      | cuda   | (32, 3, 512, 512) | 35.863 ms | 287.6 MB         | 6.8x    |
| torch_trt| dummy           | fp16      | cuda   | (40, 3, 512, 512) | 37.146 ms | 319.8 MB         | 7.4x    |
| torch_trt| dummy           | fp16      | cuda   | (48, 3, 512, 512) | 45.291 ms | 280.7 MB         | 5.0x    |
| torch_trt| dummy           | fp16      | cuda   | (56, 3, 512, 512) | 42.726 ms | 326.6 MB         | 3.8x    |
| torch_trt| dummy           | fp16      | cuda   | (64, 3, 512, 512) | 43.147 ms | 285.4 MB         | 3.9x    |
| torch    | real            | fp32      | cuda   | (32, 3, 512, 512) | 49.041 ms | 597.4 MB         | 2.2x    |
| torch    | real            | fp32      | cuda   | (40, 3, 512, 512) | 58.370 ms | 585.5 MB         | 1.1x    |
| torch    | real            | fp32      | cuda   | (48, 3, 512, 512) | 62.116 ms | 589.9 MB         | 0.7x    |
| torch    | real            | fp32      | cuda   | (56, 3, 512, 512) | 66.309 ms | 605.4 MB         | 0.4x    |
| torch    | real            | fp32      | cuda   | (64, 3, 512, 512) | 70.744 ms | 577.3 MB         | -1.3x   |
| torch    | dummy           | fp32      | cuda   | (32, 3, 512, 512) | 51.453 ms | 589.9 MB         | 2.1x    |
| torch    | dummy           | fp32      | cuda   | (40, 3, 512, 512) | 44.869 ms | 583.6 MB         | 4.6x    |
| torch    | dummy           | fp32      | cuda   | (48, 3, 512, 512) | 50.849 ms | 597.4 MB         | 1.8x    |
| torch    | dummy           | fp32      | cuda   | (56, 3, 512, 512) | 56.904 ms | 573.4 MB         | 0.8x    |
| torch    | dummy           | fp32      | cuda   | (64, 3, 512, 512) | 57.506 ms | 610.6 MB         | 1.4x    |
| onnx     | real            | fp32      | cuda   | (32, 3, 512, 512) | 36.752 ms | 158.6 MB         | 7.0x    |
| onnx     | real            | fp32      | cuda   | (40, 3, 512, 512) | 42.883 ms | 166.5 MB         | 3.4x    |
| onnx     | real            | fp32      | cuda   | (48, 3, 512, 512) | 46.565 ms | 171.1 MB         | 2.2x    |
| onnx     | real            | fp32      | cuda   | (56, 3, 512, 512) | 47.122 ms | 146.5 MB         | 2.1x    |
| onnx     | real            | fp32      | cuda   | (64, 3, 512, 512) | 58.050 ms | 177.0 MB         | -0.3x   |
| onnx     | real            | fp32      | cpu    | (32, 3, 512, 512) | 192.272 ms | 456.2 MB        | -0.7x   |
| onnx     | real            | fp32      | cpu    | (40, 3, 512, 512) | 230.900 ms | 481.6 MB        | -0.5x   |
| onnx     | real            | fp32      | cpu    | (48, 3, 512, 512) | 272.591 ms | 465.9 MB        | -0.7x   |
| onnx     | real            | fp32      | cpu    | (56, 3, 512, 512) | 315.213 ms | 478.5 MB        | -0.6x   |
| onnx     | real            | fp32      | cpu    | (64, 3, 512, 512) | 355.569 ms | 466.6 MB        | -0.7x   |
| onnx     | dummy           | fp32      | cuda   | (32, 3, 512, 512) | 38.864 ms | 174.9 MB         | 5.9x    |
| onnx     | dummy           | fp32      | cuda   | (40, 3, 512, 512) | 44.350 ms | 159.7 MB         | 4.4x    |
| onnx     | dummy           | fp32      | cuda   | (48, 3, 512, 512) | 49.661 ms | 173.5 MB         | 3.1x    |
| onnx     | dummy           | fp32      | cuda   | (56, 3, 512, 512) | 45.247 ms | 164.7 MB         | 1.0x    |
| onnx     | dummy           | fp32      | cuda   | (64, 3, 512, 512) | 50.936 ms | 144.2 MB         | 0.4x    |
| onnx     | dummy           | fp32      | cpu    | (32, 3, 512, 512) | 187.233 ms | 136.3 MB        | -0.1x   |
| onnx     | dummy           | fp32      | cpu    | (40, 3, 512, 512) | 230.421 ms | 120.1 MB        | -0.6x   |
| onnx     | dummy           | fp32      | cpu    | (48, 3, 512, 512) | 269.722 ms | 123.9 MB        | -0.6x   |
| onnx     | dummy           | fp32      | cpu    | (56, 3, 512, 512) | 308.837 ms | 144.9 MB        | -1.1x   |
| onnx     | dummy           | fp32      | cpu    | (64, 3, 512, 512) | 347.283 ms | 126.3 MB        | -0.9x   |
| torch_trt| real            | fp32      | cuda   | (32, 3, 512, 512) | 30.654 ms | 307.3 MB         | 8.7x    |
| torch_trt| real            | fp32      | cuda   | (40, 3, 512, 512) | 36.028 ms | 303.0 MB         | 5.8x    |
| torch_trt| real            | fp32      | cuda   | (48, 3, 512, 512) | 41.984 ms | 296.6 MB         | 4.7x    |
| torch_trt| real            | fp32      | cuda   | (56, 3, 512, 512) | 40.394 ms | 303.6 MB         | 3.3x    |
| torch_trt| real            | fp32      | cuda   | (64, 3, 512, 512) | 45.215 ms | 314.1 MB         | 2.9x    |
| torch_trt| dummy           | fp32      | cuda   | (32, 3, 512, 512) | 36.259 ms | 316.5 MB         | 8.9x    |
| torch_trt| dummy           | fp32      | cuda   | (40, 3, 512, 512) | 40.325 ms | 315.8 MB         | 4.9x    |
| torch_trt| dummy           | fp32      | cuda   | (48, 3, 512, 512) | 39.851 ms | 314.6 MB         | 3.6x    |
| torch_trt| dummy           | fp32      | cuda   | (56, 3, 512, 512) | 46.838 ms | 323.7 MB         | 3.2x    |
| torch_trt| dummy           | fp32      | cuda   | (64, 3, 512, 512) | 47.221 ms | 304.4 MB         | 0.9x    |

## 4. Обсуждение

### Выводы о выборе оптимального подхода:

- Для задач с максимальной требовательностью к скорости оптимально использовать PyTorch с fp16 на GPU, либо Torch-TensorRT, который предлагает более эффективное использование памяти при близкой производительности.
- Если ограничения по памяти критичны, стоит рассмотреть ONNX с fp16, принимая снижение скорости.
- Использование fp32 оправдано, если точность важнее скорости и ресурсов, но в большинстве production-задач fp16 предпочтительнее.

### Рекомендации для production использования:

- Использовать PyTorch с fp16 на CUDA для максимальной производительности и эффективности.
- Для среды с ограниченной памятью — Torch-TensorRT, поскольку он существенно снижает потребление GPU-памяти при сохранении высокой скорости.
- Внедрять ONNX-модели при необходимости легковесного решения с ограниченным ресурсом памяти, особенно если скорость не критична.
- Избегать запуска моделей на CPU в production для задач с высокими требованиями к скорости.

## 5. Заключение

### Выводы:
- torch-tensorrt показывает лучшие результаты на CUDA при любом размере.
- Увеличение размера изображения пропорционально уменьшает скорость и увеличивает память.
- При использовании ONNX наблюдается низкое потребление памяти, но более медленный инференс по сравнению с torch/torch-tensorrt.
- FP16 дает значительный прирост производительности по сравнению с FP32.
- CPU-инференс для ONNX существенно проигрывает по скорости, особенно при увеличении размера изображений.