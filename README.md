# House Price Prediction

ML-проект для предсказания цен на дома на основе датасета Kaggle ["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Описание

Этот проект реализует end-to-end ML-пайплайн для предсказания цен на недвижимость:
- Предобработка данных (заполнение пропусков, кодирование категориальных признаков)
- Обучение модели Random Forest
- Кросс-валидация и оценка качества модели
- Предсказания для тестового набора данных

## Результаты

- **CV RMSE**: ~29,903 ± 3,858
- **Hold-out RMSE**: ~28,555

## Структура проекта

```
Houses/
├── data/
│   └── raw/              # Исходные данные (train.csv, test.csv) - не коммитятся в Git
├── models/               # Сохранённые модели - не коммитятся в Git
├── notebooks/            # Jupyter ноутбуки для EDA и экспериментов
│   └── 01_baseline.ipynb # Бейзлайн модель с визуализацией
├── src/                  # Исходный код
│   └── train_baseline.py # Скрипт обучения бейзлайн модели
├── reports/              # Графики и отчёты (опционально)
├── .gitignore           # Игнорируемые файлы
├── requirements.txt     # Зависимости Python
└── README.md           # Этот файл
```

## Требования

- Python 3.8+
- Установленные зависимости из `requirements.txt`

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/Houses.git
cd Houses
```

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
```

3. Активируйте виртуальное окружение:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

4. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Подготовка данных

Скачайте датасет с Kaggle:
- Перейдите на страницу соревнования: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- Примите правила соревнования
- Скачайте `train.csv` и `test.csv`
- Поместите файлы в папку `data/raw/`:
  - `data/raw/train.csv`
  - `data/raw/test.csv`

### 2. Обучение модели

Запустите скрипт обучения:
```bash
python src/train_baseline.py
```

Скрипт:
- Загрузит данные
- Создаст и обучит модель
- Выполнит кросс-валидацию
- Сохранит модель в `models/baseline_random_forest.pkl`
- Выведет метрики качества

### 3. Работа с ноутбуком

Откройте Jupyter ноутбук для детального анализа:
```bash
jupyter notebook notebooks/01_baseline.ipynb
```

Ноутбук включает:
- Визуализацию данных
- Пошаговое обучение модели
- Предсказания для тестового набора

## Использование

### Обучение модели

```bash
python src/train_baseline.py
```

### Загрузка модели и предсказания

```python
import joblib
import pandas as pd

# Загрузите модель
pipe = joblib.load("models/baseline_random_forest.pkl")

# Загрузите тестовые данные
test_df = pd.read_csv("data/raw/test.csv")

# Сделайте предсказания
predictions = pipe.predict(test_df)

# Создайте submission файл для Kaggle
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": predictions
})
submission.to_csv("submission.csv", index=False)
```

## Особенности реализации

- **End-to-end пайплайн**: Все этапы (предобработка → обучение → предсказание) объединены в один Pipeline
- **Автоматическая обработка**: Pipeline автоматически обрабатывает числовые и категориальные признаки
- **Защита от утечек данных**: Предобработка встроена в пайплайн, предотвращая утечки данных
- **Детальное логирование**: Пошаговый вывод процесса обучения для понимания работы

## Улучшения для будущей работы

- [ ] Feature engineering (создание новых признаков)
- [ ] Гиперпараметр тюнинг
- [ ] Попробовать другие модели (XGBoost, LightGBM)
- [ ] Улучшенная обработка выбросов
- [ ] Ансамбли моделей
- [ ] Автоматизация pipeline через MLflow

## Технологии

- Python 3.8+
- pandas - работа с данными
- scikit-learn - машинное обучение
- matplotlib/seaborn - визуализация
- joblib - сохранение моделей
