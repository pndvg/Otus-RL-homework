# Otus-RL-homework

Домашние задания по Reinforcement Learning для курса Otus.

## Структура проекта

```
Otus-RL-homework/
├── src/                    # Исходный код
│   ├── agents/            # Реализации RL агентов
│   ├── environments/      # Пользовательские окружения
│   └── utils/             # Вспомогательные функции
├── notebooks/             # Jupyter ноутбуки для экспериментов
├── tests/                 # Юнит-тесты
├── configs/              # Конфигурационные файлы (гиперпараметры и т.д.)
├── results/              # Результаты обучения
│   ├── logs/             # Логи обучения
│   ├── models/           # Сохраненные модели
│   └── plots/            # Визуализации и графики
├── docs/                 # Документация
├── pyproject.toml        # Зависимости и конфигурация проекта
└── README.md            # Этот файл
```

## Установка

Проект использует [uv](https://github.com/astral-sh/uv) для управления пакетами Python.

### Требования

- Python 3.13+
- Менеджер пакетов uv

### Инструкция по установке

1. Клонировать репозиторий:
```bash
git clone <repository-url>
cd Otus-RL-homework
```

2. Установить зависимости с помощью uv:
```bash
uv sync
```

Это создаст виртуальное окружение в `.venv/` и установит все необходимые пакеты.

## Использование

### Запуск Python скриптов

Используйте `uv run` для выполнения Python скриптов в виртуальном окружении:

```bash
uv run python src/your_script.py
```

### Запуск Jupyter ноутбуков

Запустить Jupyter Lab или Jupyter Notebook:

```bash
uv run jupyter lab
# или
uv run jupyter notebook
```

### Запуск тестов

Выполнить тесты с помощью pytest:

```bash
uv run pytest
```

### Форматирование и линтинг кода

Форматировать код с помощью ruff:

```bash
uv run ruff format .
```

Проверить код с помощью ruff:

```bash
uv run ruff check .
```

## Зависимости

Основные зависимости:
- `gymnasium` - фреймворк для RL окружений
- `numpy` - численные вычисления
- `matplotlib` - визуализация и графики
- `jupyter` - интерактивные ноутбуки

Зависимости для разработки:
- `pytest` - фреймворк для тестирования
- `ruff` - линтер и форматтер

Полный список зависимостей см. в [pyproject.toml](pyproject.toml).

## Добавление новых зависимостей

Чтобы добавить новый пакет:

```bash
uv add package-name
```

Для зависимостей разработки:

```bash
uv add --dev package-name
```
