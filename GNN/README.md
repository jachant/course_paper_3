<p align="center">
	<img src="https://github.com/daiquocnguyen/GNN-ReGVD/blob/master/logo.png" width="125">
</p>

# GNN-ReGVD + LoRA + FAISS: Гибридная система детекции уязвимостей

Расширение архитектуры [ReGVD](https://arxiv.org/abs/2110.07317) (ICSE 2022) двумя компонентами: **LoRA-адаптерами** для дешёвого дообучения и **FAISS-индексом** для мгновенной реакции на новые уязвимости без переобучения модели.

---

## Содержание

1. [Общая архитектура](#общая-архитектура)
2. [Полный цикл работы](#полный-цикл-работы)
3. [Обоснование стека](#обоснование-стека)
4. [Описание компонентов](#описание-компонентов)
5. [Команды запуска](#команды-запуска)
6. [Структура проекта](#структура-проекта)
7. [Гиперпараметры](#гиперпараметры)

---

## Общая архитектура

```
                    Исходный код (функция C/C++)
                             │
                             ▼
              ┌──────────────────────────────┐
              │  GraphCodeBERT (frozen)       │
              │  + LoRA adapters (r=8, α=16)  │
              │  ~1M trainable / 125M total   │
              └──────────────┬───────────────┘
                             │ word embeddings (768-d)
                             ▼
              ┌──────────────────────────────┐
              │  Graph Construction           │
              │  (sliding window → adj matrix)│
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  GNN (ReGCN / ReGGNN)         │
              │  + residual connections       │
              │  + soft attention pooling      │
              └──────────┬───────────────────┘
                         │ graph-level embedding
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
   ┌─────────────────┐   ┌──────────────────┐
   │  Classification  │   │  Embedding Head   │
   │  Head            │   │  (512-d, L2-norm) │
   │  → P(vulnerable) │   └────────┬─────────┘
   └────────┬────────┘            │
            │                     ▼
            │           ┌──────────────────┐
            │           │   FAISS Index     │
            │           │   (top-K поиск)   │
            │           └────────┬─────────┘
            │                    │
            └────────┬───────────┘
                     │
                     ▼
            ┌────────────────────┐
            │  Гибридный скор     │
            │  β·P + (1-β)·FAISS  │
            └────────────────────┘
```

---

## Полный цикл работы

### Фаза 1 — Начальное обучение

Обучаем модель с двумя функциями потерь одновременно:

```
Total Loss = α · BCE_loss + (1-α) · Contrastive_loss
```

**BCE loss** учит классификатор отличать уязвимый код от безопасного. **Contrastive loss** учит embedding head строить такое пространство, где уязвимые сниппеты кластеризуются вместе и отделяются от безопасных. Это пространство потом используется FAISS для nearest-neighbor поиска.

После обучения автоматически строится FAISS-индекс: все обучающие примеры прогоняются через модель, 512-мерные эмбеддинги сохраняются вместе с метаданными (метка, индекс, сниппет кода).

**Что обучается:**
- LoRA-адаптеры в GraphCodeBERT (~1M параметров) — query и value проекции каждого attention head
- GNN-слои (~300K параметров) — графовые свёртки
- Classification head (~50K) — бинарная классификация
- Embedding head (~300K) — проекция для FAISS

**Что заморожено:**
- Основные 125M параметров GraphCodeBERT — не изменяются вообще

### Фаза 2 — Инференс

Новый код проходит через всю модель и получает два скора:

1. **cls_prob** — вероятность уязвимости от классификатора (0..1)
2. **faiss_score** — доля уязвимых среди K ближайших соседей в FAISS (0..1)

Финальный скор: `hybrid = β · cls_prob + (1-β) · faiss_score`, где β подбирается на валидации (по умолчанию 0.6).

### Фаза 3 — Дообучение на новых данных

Два режима обновления для разных ситуаций:

**Режим A — Горячее обновление (Hot Update)**

Когда: появились новые размеченные примеры (десятки-сотни), нужна мгновенная реакция.

Что происходит: новые примеры прогоняются через замороженную модель, их эмбеддинги добавляются в FAISS-индекс. Модель не трогается. Время: секунды.

```
Новые данные → Замороженная модель → Эмбеддинги → faiss_index.add() → Готово
```

**Режим B — Периодическое дообучение (LoRA Fine-tune)**

Когда: накопилось 500+ новых примеров, хочется улучшить сами представления.

Что происходит: LoRA-адаптеры, GNN и обе головы дообучаются на новых данных (с подмешиванием 20% старых для борьбы с catastrophic forgetting). Затем FAISS-индекс перестраивается с нуля с обновлёнными эмбеддингами. Время: минуты-часы.

```
Новые + 20% старых данных → LoRA fine-tune → Rebuild FAISS → Готово
```

---

## Обоснование стека

### Почему LoRA, а не полное fine-tuning

GraphCodeBERT содержит 125M параметров. Полное обучение всех весов требует значительной GPU-памяти и данных. LoRA (Low-Rank Adaptation) замораживает оригинальные веса и добавляет маленькие обучаемые матрицы (rank=8) в attention-слои. Итого ~1% обучаемых параметров при сравнимом качестве. Главное преимущество: можно хранить несколько LoRA-адаптеров под разные задачи (один для C/C++, другой для Java) и переключаться без дублирования базовой модели.

### Почему собственная реализация LoRA, а не peft

Библиотека `peft` от HuggingFace — стандартный инструмент для LoRA, но она рассчитана на чистые transformer-модели. В нашей архитектуре GraphCodeBERT используется нестандартно: его word embeddings идут напрямую в графовый конвейер, а не через обычный forward pass. Собственный `LoRALinear` даёт полный контроль над тем, какие слои оборачиваются, и не конфликтует с GNN-частью модели.

### Почему FAISS

FAISS (Facebook AI Similarity Search) — индустриальный стандарт для approximate nearest neighbor search. Ключевые свойства для нашей задачи:

- **Скорость**: поиск по 100K+ векторов за миллисекунды
- **Масштабируемость**: поддержка IVF-индексов для миллионов векторов
- **Hot update**: `index.add()` добавляет новые векторы без перестроения
- **Inner product search**: после L2-нормализации эквивалентен cosine similarity

Для датасетов до 100K используется `IndexFlatIP` (точный поиск). Для более крупных — `IndexIVFFlat` (приближённый, но в 10-100x быстрее).

### Почему Supervised Contrastive Loss

Стандартный triplet loss работает с тройками (anchor, positive, negative). Supervised Contrastive Loss (SupCon) обобщает это на весь батч: каждый пример сравнивается со всеми другими, а не только с одной парой. Это даёт более стабильные градиенты и лучшие эмбеддинги при том же размере батча. Реализация включает online hard mining для автоматического выбора самых сложных пар.

### Почему гибридный скор, а не один из двух

Классификатор хорошо обобщает паттерны из обучающих данных, но плохо реагирует на принципиально новые типы уязвимостей. FAISS, наоборот, мгновенно реагирует на новые примеры через горячее обновление, но зависит от качества покрытия базы. Комбинация двух скоров через взвешенное среднее позволяет компенсировать слабости каждого подхода. Параметр β подбирается на валидации через grid search.

### Почему replay при дообучении

Catastrophic forgetting — фундаментальная проблема нейросетей: при обучении на новых данных модель «забывает» старые паттерны. Replay ratio=0.2 означает, что при дообучении 20% батча составляют случайные примеры из оригинальных данных. Это простой и эффективный способ сохранить качество на старых паттернах.

---

## Описание компонентов

### model.py — Модели

- **LoRALinear**: обёртка над `nn.Linear`, добавляющая low-rank матрицы A и B. Инициализация: A — Kaiming uniform, B — нули (LoRA стартует как identity).
- **apply_lora_to_model()**: применяет LoRA к query/value слоям GraphCodeBERT. Замораживает все остальные параметры.
- **EmbeddingHead**: двухслойный проектор `Linear → ReLU → Dropout → Linear → L2-norm`. Выход: 512-мерный единичный вектор для FAISS.
- **GNNReGVD**: расширенная версия оригинальной модели. Forward возвращает `(cls_loss, prob, embeddings)` при обучении и `(prob, embeddings)` при инференсе. Метод `get_embedding()` — для извлечения эмбеддингов без классификации.

### losses.py — Функции потерь

- **SupervisedContrastiveLoss**: строит матрицу cosine similarity для всего батча, вычисляет log-softmax по positive парам. Temperature=0.07 контролирует «остроту» распределения.
- **TripletMarginLossWithMining**: online hard mining — для каждого anchor находит самый далёкий positive и самый близкий negative. Margin=0.3.

### faiss_index.py — FAISS менеджер

- **build_index()**: строит индекс с нуля. L2-нормализует векторы для cosine similarity через inner product.
- **add()**: горячее обновление — добавляет новые векторы без перестроения.
- **search()**: top-K поиск с возвратом метаданных.
- **compute_faiss_score()**: доля уязвимых среди K ближайших — это и есть FAISS-скор для гибридного решения.
- **save() / load()**: сериализация индекса + метаданных + конфига на диск.

### run.py — Обучение

Расширенный оригинальный скрипт. Добавлено: dual loss (BCE + contrastive), оптимизация только trainable параметров, автоматическое построение FAISS-индекса после обучения. Полностью обратно совместим: без флагов `--use_lora` и `--use_faiss` работает как оригинал.

### run_finetune.py — Дообучение

Два режима в одном скрипте:

- `--mode hot_update`: загружает модель и FAISS, прогоняет новые данные, делает `faiss_manager.add()`.
- `--mode lora_finetune`: дообучает LoRA+GNN+heads, затем перестраивает FAISS с нуля.

### inference.py — Инференс

Класс `HybridPredictor` с методами `predict_single()` и `predict_batch()`. Функция `find_optimal_beta()` делает grid search по β на валидации.

---

## Команды запуска

### Установка зависимостей

```bash
pip install torch transformers faiss-cpu scipy numpy
# Для GPU: pip install faiss-gpu вместо faiss-cpu
```

### 1. Полное обучение с LoRA + FAISS

```bash
cd code

python run.py \
  --output_dir=./saved_models/lora_faiss \
  --model_type=roberta \
  --tokenizer_name=microsoft/graphcodebert-base \
  --model_name_or_path=microsoft/graphcodebert-base \
  --do_train --do_eval --do_test --evaluate_during_training \
  --train_data_file=../dataset/train.jsonl \
  --eval_data_file=../dataset/valid.jsonl \
  --test_data_file=../dataset/test.jsonl \
  --block_size 400 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --max_grad_norm 1.0 \
  --gnn ReGCN \
  --learning_rate 5e-4 \
  --epoch 100 \
  --hidden_size 128 \
  --num_GNN_layers 2 \
  --format uni \
  --window_size 5 \
  --seed 123456 \
  --use_lora --lora_rank 8 --lora_alpha 16 \
  --use_faiss --embed_dim 512 \
  --contrastive_weight 0.3 \
  --contrastive_loss supcon
```

### 2. Режим A — Горячее обновление (секунды, 0 retraining)

```bash
python run_finetune.py \
  --mode hot_update \
  --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
  --new_data ../dataset/new_samples.jsonl \
  --faiss_dir ./saved_models/lora_faiss/faiss_index \
  --use_lora --use_faiss --embed_dim 512 \
  --model_name_or_path=microsoft/graphcodebert-base \
  --tokenizer_name=microsoft/graphcodebert-base
```

### 3. Режим B — Периодическое дообучение (LoRA fine-tune + FAISS rebuild)

```bash
python run_finetune.py \
  --mode lora_finetune \
  --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
  --new_data ../dataset/new_samples.jsonl \
  --train_data_file ../dataset/train.jsonl \
  --eval_data_file ../dataset/valid.jsonl \
  --output_dir ./saved_models/lora_faiss_v2 \
  --use_lora --use_faiss --embed_dim 512 \
  --model_name_or_path=microsoft/graphcodebert-base \
  --tokenizer_name=microsoft/graphcodebert-base \
  --epoch 10 \
  --learning_rate 1e-4 \
  --replay_ratio 0.2
```

### 4. Гибридный инференс с автоподбором β

```bash
python inference.py \
  --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
  --faiss_dir ./saved_models/lora_faiss/faiss_index \
  --input_file ../dataset/test.jsonl \
  --output_file ./predictions_hybrid.json \
  --use_lora --use_faiss --embed_dim 512 \
  --model_name_or_path=microsoft/graphcodebert-base \
  --tokenizer_name=microsoft/graphcodebert-base \
  --beta 0.6 \
  --top_k 5 \
  --find_beta
```

### 5. Обучение без LoRA/FAISS (оригинальный режим, обратная совместимость)

```bash
python run.py \
  --output_dir=./saved_models/baseline \
  --model_type=roberta \
  --tokenizer_name=microsoft/graphcodebert-base \
  --model_name_or_path=microsoft/graphcodebert-base \
  --do_train --do_eval --do_test --evaluate_during_training \
  --train_data_file=../dataset/train.jsonl \
  --eval_data_file=../dataset/valid.jsonl \
  --test_data_file=../dataset/test.jsonl \
  --block_size 400 --train_batch_size 128 --gnn ReGCN \
  --learning_rate 5e-4 --epoch 100 --hidden_size 128 \
  --seed 123456
```

---

## Структура проекта

```
GNN-ReGVD/
├── code/
│   ├── model.py               # GNNReGVD + LoRA + EmbeddingHead
│   ├── modelGNN_updates.py    # ReGCN, ReGGNN, graph construction
│   ├── utils.py               # Preprocessing utilities
│   ├── losses.py              # SupCon и Triplet losses
│   ├── faiss_index.py         # FAISS index manager
│   ├── run.py                 # Полное обучение (dual loss)
│   ├── run_finetune.py        # Hot update (A) и LoRA fine-tune (B)
│   └── inference.py           # Гибридный инференс + подбор β
├── dataset/
│   ├── train.jsonl            # 21,854 примера
│   ├── valid.jsonl            # 2,732 примера
│   └── test.jsonl             # 2,732 примера
├── evaluator/
│   └── evaluator.py
└── README.md
```

---

## Гиперпараметры

| Параметр | Значение | Обоснование |
|---|---|---|
| `lora_rank` | 8 | Баланс между ёмкостью и числом параметров. Rank 4-16 — стандартный диапазон |
| `lora_alpha` | 16 | Scaling = alpha/rank = 2.0. Стандартное значение из оригинальной статьи LoRA |
| `embed_dim` | 512 | Достаточно для различения паттернов кода. 256 — слишком сжато, 768 — избыточно |
| `contrastive_weight` | 0.3 | 70% BCE + 30% contrastive. Классификация — основная задача, contrastive — вспомогательная |
| `temperature` | 0.07 | Стандарт для SupCon. Ниже — более «острое» распределение, выше — более размытое |
| `beta` | 0.6 | Классификатор получает больший вес. Подбирается на валидации через `--find_beta` |
| `top_k` | 5 | Число соседей для FAISS-скора. 3-10 — разумный диапазон |
| `replay_ratio` | 0.2 | 20% старых данных при дообучении. Достаточно против catastrophic forgetting |
| `learning_rate` (finetune) | 1e-4 | В 5x ниже, чем при начальном обучении — избегаем разрушения выученных представлений |

---

## Формат входных данных

JSONL, по одному JSON-объекту на строку:

```json
{"func": "int vulnerable_function(char *buf) { strcpy(dest, buf); return 0; }", "target": 1, "idx": 0}
{"func": "int safe_function(int a, int b) { return a + b; }", "target": 0, "idx": 1}
```

- `func` — исходный код функции (строка)
- `target` — метка: 1 = уязвимый, 0 = безопасный
- `idx` — уникальный идентификатор примера

---

## Cite

Оригинальная статья ReGVD:

```bibtex
@inproceedings{NguyenReGVD,
    author={Van-Anh Nguyen and Dai Quoc Nguyen and Van Nguyen and Trung Le and Quan Hung Tran and Dinh Phung},
    title={ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection},
    booktitle={Proceedings of the 44th International Conference on Software Engineering Companion (ICSE '22 Companion)},
    year={2022}
}
```

## License

As a free open-source implementation, ReGVD is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
