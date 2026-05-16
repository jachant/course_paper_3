# vuln-triage

Автоматический триаж уязвимостей, обнаруженных SAST-инструментами, с использованием CodeQL и LLM.

На вход принимает SARIF-отчёт и путь к исходному коду, на выходе — JSON-отчёт с вердиктом по каждому нахождению: **true positive**, **false positive** или **uncertain**.

## Как это работает

Пайплайн состоит из четырёх этапов:

1. **Парсинг SARIF** — извлечение нахождений (rule, severity, location, codeFlows) из стандартного SARIF v2.1.0 отчёта.
2. **Извлечение кода** — чтение фрагмента исходного кода вокруг уязвимой строки с маркировкой `>>>`.
3. **CodeQL-анализ** — создание базы данных CodeQL и выполнение шести QL-запросов: пути taint-потоков, санитайзеры/гарды, охватывающая функция, граф вызовов (callers/callees), классификация sink'а, классификация источника данных.
4. **LLM-инференс** — отправка собранного контекста (SARIF-метаданные + код + CodeQL) в LLM для классификации.

Если CodeQL недоступен или база данных не строится, пайплайн продолжает работу только на основе кода (graceful degradation).

## Требования

- Python 3.10+
- [CodeQL CLI](https://github.com/github/codeql-cli-binaries) + пак `codeql/python-all`
- API-ключ LLM-провайдера (Anthropic, OpenAI и т.д.)

## Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd vuln-triage

# 2. Установить Python-зависимости
pip install -e .

# 3. Установить CodeQL CLI
#    Скачать с https://github.com/github/codeql-cli-binaries/releases
#    и распаковать, например, в ~/codeql/

# 4. Скачать пак codeql/python-all (если не установлен)
codeql pack download codeql/python-all

# 5. Настроить путь к CodeQL и API-ключ
cp .env.example .env
# Отредактировать .env — вставить свой ключ
```

Прописать путь к бинарнику CodeQL в `config.yaml`:

```yaml
codeql:
  codeql_path: "/path/to/codeql/codeql"
```

## Настройка API-ключа

Проект использует [litellm](https://docs.litellm.ai/) для абстракции LLM-провайдеров. Инференс выполняется через [OpenRouter](https://openrouter.ai/), что даёт доступ к Claude, GPT-4o и другим моделям через единый ключ.

**Вариант 1 — файл `.env`** (рекомендуется):

```bash
cp .env.example .env
```

Откройте `.env` и вставьте свой ключ OpenRouter (получить: https://openrouter.ai/keys):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Затем экспортируйте перед запуском:

```bash
export $(cat .env | grep -v '^#' | xargs)
```

**Вариант 2 — напрямую в терминале:**

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Модель задаётся в `config.yaml` (поле `llm.model`) в формате `openrouter/<провайдер>/<модель>`. По умолчанию — `openrouter/anthropic/claude-sonnet-4-6`.

## Использование

```bash
# Базовый запуск
vuln-triage triage <путь-к-sarif-файлу> <путь-к-исходному-коду>

# С явным указанием конфига и выходного файла
vuln-triage triage report.sarif.json ./src --config config.yaml --output results.json

# Без CodeQL (только код + LLM)
vuln-triage triage report.sarif.json ./src --skip-graph

# С другой моделью
vuln-triage triage report.sarif.json ./src --model openrouter/openai/gpt-4o

# Подробный вывод
vuln-triage triage report.sarif.json ./src --verbose

# Оценка точности по размеченному датасету
vuln-triage evaluate dataset.json --output results.json
vuln-triage evaluate dataset.json --limit 50 --skip-graph
```

## Конфигурация

Файл `config.yaml`:

```yaml
codeql:
  codeql_path: "/path/to/codeql/codeql"  # путь к бинарнику CodeQL CLI
  search_path: ""                          # путь к QL-пакам (авто-определяется, если пусто)
  language: "python"                       # язык анализа
  db_timeout: 600                          # таймаут создания базы данных (сек.)
  query_timeout: 300                       # таймаут на один запрос (сек.)

llm:
  model: "openrouter/anthropic/claude-sonnet-4-6"  # формат: openrouter/<провайдер>/<модель>
  temperature: 0.1             # низкая температура для детерминизма
  max_tokens: 2048
  max_concurrent: 5            # макс. параллельных LLM-запросов

extraction:
  context_lines: 30            # строк контекста вокруг нахождения
  max_snippet_lines: 100       # макс. длина фрагмента

pipeline:
  max_concurrent_findings: 10  # макс. параллельно обрабатываемых нахождений
  fallback_to_ast: true        # продолжить без CodeQL, если он недоступен
```

## Формат выходного отчёта

```json
{
  "report": {
    "generated_at": "2026-05-04T12:00:00+00:00",
    "sarif_file": "report.sarif.json",
    "source_path": "/path/to/src",
    "llm_model": "claude-sonnet-4-6",
    "total_findings": 15,
    "true_positives": 4,
    "false_positives": 9,
    "uncertain": 2,
    "errors": 0
  },
  "findings": [
    {
      "rule_id": "B301",
      "file": "app/db.py",
      "line": 45,
      "severity": "high",
      "tool_message": "Use of insecure function",
      "verdict": "false_positive",
      "confidence": 0.92,
      "reasoning": "Параметризованный запрос через ORM..."
    }
  ]
}
```

## Архитектура

```
src/vuln_triage/
├── cli.py              # точка входа CLI (Typer): команды triage и evaluate
├── config.py           # Pydantic-модели конфигурации
├── pipeline.py         # оркестратор пайплайна
├── eval_pipeline.py    # пайплайн оценки точности по датасету
├── sarif/
│   ├── parser.py       # парсинг SARIF v2.1.0
│   └── models.py       # датаклассы Finding, TriageVerdict, TriageReport
├── code/
│   └── extractor.py    # извлечение фрагментов кода
├── codeql/
│   ├── codeql_client.py   # низкоуровневый интерфейс к CodeQL CLI
│   │                      # (create database, run query, decode BQRS→CSV)
│   ├── codeql_analyzer.py # высокоуровневый анализатор: запускает 6 запросов
│   │                      # и собирает контекст для LLM-промпта
│   └── queries.py         # QL-шаблоны запросов (taint paths, sanitizers,
│                          # scope, call graph, sink/source classification)
├── llm/
│   ├── client.py       # LLM-клиент через litellm
│   └── prompts.py      # системный и пользовательский промпты
└── report/
    └── generator.py    # генерация JSON-отчёта
```

### CodeQL-запросы

Для каждого нахождения `CodeQLAnalyzer` поочерёдно выполняет:

1. **Taint flow paths** — полные цепочки source → sink через `TaintTracking::Global`; использует `RemoteFlowSource` как источники и область ±2 строки от нахождения как sink.
2. **Sanitizers & guards** — вызовы функций с именами, соответствующими паттерну `escap|sanitiz|validat|...` внутри охватывающей функции.
3. **Enclosing scope** — самая узкая функция/метод, содержащая строку нахождения.
4. **Call graph** — callers (кто вызывает охватывающую функцию) и callees (что вызывается внутри неё).
5. **Sink classification** — тип sink'а по CodeQL Concepts: `SqlExecution`, `CommandExecution`, `FileSystemAccess`, `CodeExecution`.
6. **Source classification** — какие `RemoteFlowSource` достигают строки нахождения.

## Зависимости

- **typer** — CLI-фреймворк
- **pydantic** / **pydantic-settings** — типизированная конфигурация
- **litellm** (1.80–1.81) — унифицированный интерфейс к LLM-провайдерам (инференс через OpenRouter)
- **pyyaml** — чтение config.yaml
- **rich** — форматированный вывод в терминал
- **CodeQL CLI** (внешняя зависимость) — статический анализ и построение графа

## Тестирование

```bash
pip install -e ".[dev]"
pytest
```

## Лицензия

MIT
