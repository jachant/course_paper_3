"""
Диагностика GPU для GNN-ReGVD.
Запуск: python check_gpu.py
"""

import sys
print(f"Python: {sys.version}")

# ── 1. PyTorch и CUDA ──
try:
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"Число GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_memory / 1024**2
            free_mb = torch.cuda.mem_get_info(i)[0] / 1024**2
            used_mb = total_mb - free_mb
            print(f"\n  GPU {i}: {props.name}")
            print(f"    VRAM: {total_mb:.0f} MB всего, {free_mb:.0f} MB свободно, {used_mb:.0f} MB занято")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    SM count: {props.multi_processor_count}")

        # Быстрый тест — прогоняем тензор через GPU
        device = torch.device("cuda:0")
        a = torch.randn(1000, 1000, device=device)
        b = torch.matmul(a, a)
        print(f"\n  Тест matmul на GPU: OK (результат shape={b.shape})")
        del a, b
        torch.cuda.empty_cache()
    else:
        print("\n  GPU не найдена. Возможные причины:")
        print("  - Не установлен CUDA-совместимый PyTorch (pip install torch --index-url https://download.pytorch.org/whl/cu121)")
        print("  - Драйвер NVIDIA не установлен или устарел (nvidia-smi)")
        print("  - Нет NVIDIA GPU в системе")
except ImportError:
    print("\nPyTorch НЕ УСТАНОВЛЕН. Установи: pip install torch")

# ── 2. nvidia-smi ──
print("\n" + "=" * 50)
print("nvidia-smi:")
import subprocess
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("  nvidia-smi вернул ошибку. Драйвер NVIDIA не установлен?")
except FileNotFoundError:
    print("  nvidia-smi не найден. Драйвер NVIDIA не установлен.")
except Exception as e:
    print(f"  Ошибка: {e}")

# ── 3. Зависимости проекта ──
print("=" * 50)
print("Зависимости проекта:")

deps = {
    "transformers": "transformers",
    "faiss (cpu)": "faiss",
    "scipy": "scipy",
    "numpy": "numpy",
    "sklearn": "sklearn",
}

for name, module in deps.items():
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  {name}: {ver}")
    except ImportError:
        print(f"  {name}: НЕ УСТАНОВЛЕН")

# ── 4. Проверка загрузки модели ──
print("\n" + "=" * 50)
print("Проверка GraphCodeBERT:")
try:
    from transformers import RobertaTokenizer, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    print(f"  Tokenizer: OK (vocab_size={tokenizer.vocab_size})")
    print(f"  Config: OK (hidden_size={config.hidden_size})")
except Exception as e:
    print(f"  Ошибка: {e}")
    print("  Модель загрузится автоматически при первом запуске (нужен интернет)")

# ── 5. Рекомендация по batch_size ──
if torch.cuda.is_available():
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print("\n" + "=" * 50)
    print("Рекомендуемые настройки:")
    if total_mb >= 16000:
        print(f"  VRAM {total_mb:.0f} MB → batch_size=128, gradient_accumulation=1")
    elif total_mb >= 8000:
        print(f"  VRAM {total_mb:.0f} MB → batch_size=32, gradient_accumulation=4")
    elif total_mb >= 4000:
        print(f"  VRAM {total_mb:.0f} MB → batch_size=8, gradient_accumulation=16")
    else:
        print(f"  VRAM {total_mb:.0f} MB → batch_size=4, gradient_accumulation=32")
