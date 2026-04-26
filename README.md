[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Сегментация городских сцен]

**Студент:** [Лоскутов Михаил Ильич]

**Группа:** [237]


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Семантическая сегментация городских сцен - присвоение каждому пикселю изображения одного из 19 семантических классов (дорога, тротуар, здания, автомобили, пешеходы и др.).

**Датасет:** CityScapes - бенчмарк для оценки алгоритмов сегментации в условиях городского вождения.
 - 5,000 изображений с детальной pixel-level аннотацией
 - Разрешение: 2048×1024 пикселей
 - 19 оцениваемых классов, сгруппированных в 8 категорий
 - Официальное разбиение: train (2,975), val (500), test (1,525)
 - больше 11Гб распакованный датасет!

**Целевая метрика:** Mean Intersection over Union (mIoU) - средняя по классам доля пересечения предсказанной и истинной области.


## Структура репозитория
<!-- Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости. -->
```
.
├── data
│   ├── raw
│   │   ├── cityscapes/
│   │   │   ├── gtFine/                 # Аннотации (маски)
│   │   │   │   ├── train/
│   │   │   │   ├── val/
│   │   │   │   └── test/
│   │   │   ├── leftImg8bit/            # Исходные изображения
│   │   │   │   ├── train/
│   │   │   │   ├── val/
│   │   │   │   └── test/
│   │   │   ├── gtFine_trainvaltest.zip
│   │   │   └── leftImg8bit_trainvaltest.zip
│   │   └── downloads/                  # Временные файлы загрузки
│   └── processed
│       ├── class_distribution.png      # График распределения классов
│       ├── samples/                    # Визуализированные примеры
│       └── integrity_report.json       # Отчёт о проверке целостности
├── models
│   ├── baseline_unet.pt                # Сохранённая baseline-модель
│   └── checkpoints/                    # Чекпоинты в процессе обучения
├── notebooks
|   ├── eda.ipynb                       # Анализ
│   └── experiments.ipynb               # Эксперименты, baseline, архитектуры, метрики
├── presentation
│   └── presentation.pdf                # Слайды для защиты
├── report
│   ├── images/                         # Изображения для отчёта
│   └── report.md                       # Финальный отчёт
├── src
│   ├── data_loader.py                  # Загрузка и парсинг CityScapes
│   ├── preprocessing.py                # Аугментация, нормализация, resize
│   ├── metrics.py                      # Расчёт IoU, mIoU, pixel accuracy
│   ├── models.py                       # Архитектуры: U-Net, DeepLabV3
│   ├── train.py                        # Цикл обучения
│   └── utils.py                        # Вспомогательные функции
├── tests
│   ├── test_data_integrity.py          # Тесты соответствия изображений и масок
│   └── test_metrics.py                 # Тесты метрик на синтетических данных
├── requirements.txt
├── download_cityscapes.py              # Скрипт авторизованной загрузки
└── README.md
```

## Запуск
На данный момент запускаются лишь ноутбуки для анализа и первых экспериментов работы моделей/лоадеров/датасетов (кастомных). Для запуска необходимо выполнить следующие команды:
```
git clone <url>
cd <repo-name>

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install cityscapesscripts
```
После этого подключить ядро ноутбуков к этому интерпретатору, а также загрузить сам датасет (инструкция ниже)

Также необходимо создать файл .env в корне проекта и добавить туда переменную:
```
DOWNLOAD_DIR='C:/absolute/path/to/data/raw/cityscapes'
```

## Данные
### Источники
| Пакет | Package ID | Описание  | Размер |
|--------|-------------|-------------|------------|
| gtFine | 1 | Аннотации высокого качества (маски) | ~2.5 ГБ |
| leftImg8bit | 3 | Исходные RGB-изображения с левой камеры | ~11 ГБ |

### Установка
Для установки необходимо зарегестрироваться на официальном сайте [CityScapes] (https://www.cityscapes-dataset.com) и ввести логин и пароль при вводе команд ниже

```
csDownload -d C:\absolute\path\to\data\raw\cityscapes gtFine_trainvaltest.zip
csDownload -d C:\absolute\path\to\data\raw\cityscapes leftImg8bit_trainvaltest.zip

# Распаковка
cd data/raw/cityscapes
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip

# Windows
# Expand-Archive .\gtFine_trainvaltest.zip .
# Expand-Archive .\leftImg8bit_trainvaltest.zip .
```

### Структура файлов

После распаковки датасет имеет иерархическую структуру по городам и сплитам:
```cityscapes/
├── gtFine/
│   └── train/
│       └── aachen/
│           ├── aachen_000000_000019_gtFine_labelIds.png
│           └── ...
└── leftImg8bit/
    └── train/
        └── aachen/
            ├── aachen_000000_000019_leftImg8bit.png
            └── ...
```

## Результаты
Результаты по CP1 представлены в README.MD в notebooks/
Здесь коротко выпишите результаты.
| Модель | [Метрика 1] | [Метрика 2] | Примечание |
|--------|-------------|-------------|------------|
| Baseline | — | — | |
| Лучшая модель | — | — | |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
