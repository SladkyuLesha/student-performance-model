# Student Performance Prediction

Проект по анализу табличных данных о студентах и построению модели машинного обучения для предсказания итоговой оценки `G3`.

Цель проекта - собрать понятный и воспроизводимый ML-пайплайн: загрузка данных, первичный анализ, preprocessing, честное разбиение train/test, обучение моделей, расчет метрик, построение графиков и сохранение результатов в `reports`.

## Структура Проекта

```text
student-performance-model/
  data/
    raw/                    # исходные CSV-файлы, хранятся локально
    processed/              # обработанные данные, если они понадобятся
  notebooks/
    01_data_overview.ipynb  # исследовательский анализ и первые эксперименты
  reports/
    figures/                # сохраненные графики
    tables/                 # сохраненные таблицы с метриками
  src/
    data_loader.py          # загрузка CSV-файлов
    preprocessing.py        # обработка данных и подготовка признаков
    splitting.py            # честное train/test-разбиение через GroupShuffleSplit
    check_preprocessing.py  # проверочный запуск подготовки данных
    metrics.py              # расчет и сборка метрик
    train_models.py         # функции обучения моделей
    visualization.py        # функции построения графиков
    run_training.py         # запуск текущего training pipeline
  .gitattributes
  .gitignore
  README.md
  requirements.txt
```

## Используемые Технологии

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

Зависимости перечислены в `requirements.txt`.

## Установка

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение в Windows PowerShell:

```bash
.venv\Scripts\activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

Основной исследовательский анализ находится в notebook:

```text
notebooks/01_data_overview.ipynb
```

Проверить загрузку, preprocessing и разбиение данных:

```bash
python -m src.check_preprocessing
```

Запустить текущий training pipeline:

```bash
python -m src.run_training
```

После запуска создаются:

```text
reports/tables/model_metrics.csv
reports/figures/r2_comparison.png
reports/figures/mae_comparison.png
reports/figures/actual_vs_predicted_gb_with_grades.png
```

## Данные

В проекте используются два CSV-файла:

- `student_data.csv` - базовый датасет на 395 строк.
- `student_data_500k.csv` - расширенный датасет на 500000 строк.

Источник базового датасета: [Student Performance Data на Kaggle](https://www.kaggle.com/datasets/devansodariya/student-performance-data).

CSV-файлы должны лежать локально в `data/raw`. Они не добавляются в Git, потому что данные могут быть большими и не должны храниться в репозитории.

Расширенный датасет содержит технический столбец `record_id`. Он исключается из признаков, потому что является идентификатором записи, а не характеристикой студента.

## Целевая Переменная

Целевая переменная:

```text
G3
```

`G3` - итоговая оценка студента. Задача проекта формулируется как регрессия: модель предсказывает числовое значение итоговой успеваемости.

В проекте сравниваются два сценария:

- с `G1` и `G2` - модель использует промежуточные оценки;
- без `G1` и `G2` - модель прогнозирует `G3` только по остальным признакам.

## Текущий Pipeline

1. `load_data` загружает CSV-файл в `pandas.DataFrame`.
2. `preprocess_data` удаляет `record_id` и кодирует категориальные признаки через `pd.get_dummies`.
3. `make_feature_sets` создает два набора признаков: с `G1/G2` и без `G1/G2`.
4. `honest_split_data` делает честное train/test-разбиение через `GroupShuffleSplit`.
5. `train_models.py` обучает `DummyRegressor`, `LinearRegression` и `HistGradientBoostingRegressor`.
6. `metrics.py` рассчитывает `MAE`, `MSE`, `RMSE` и `R2`, а затем собирает строки для итоговой таблицы.
7. `visualization.py` строит графики сравнения моделей и факт/прогноз.
8. `run_training.py` запускает весь текущий пайплайн и сохраняет результаты.

## Модели

### DummyRegressor

Baseline-модель:

```python
DummyRegressor(strategy="mean")
```

Она всегда предсказывает среднее значение `G3`. Нужна как точка отсчета: полезная модель должна быть лучше простого среднего.

### LinearRegression

Линейная регрессия используется как простая интерпретируемая модель для сравнения с baseline и бустингом.

### HistGradientBoostingRegressor

Основная модель проекта для табличных данных. Используется градиентный бустинг из scikit-learn.

Текущие параметры последнего запуска:

```text
max_iter=20000
learning_rate=0.03
min_samples_leaf=70
l2_regularization=2.0
n_iter_no_change=60
random_state=42
```

## Метрики

### MAE

```text
MAE = mean(abs(y_true - y_pred))
```

Средняя абсолютная ошибка. Показывает, на сколько баллов в среднем ошибается модель.

### MSE

```text
MSE = mean((y_true - y_pred)^2)
```

Средняя квадратичная ошибка. Сильнее штрафует большие ошибки.

### RMSE

```text
RMSE = sqrt(MSE)
```

Корень из MSE. Измеряется в тех же единицах, что и `G3`.

### R2

```text
R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
```

Показывает, какую долю разброса целевой переменной объясняет модель. Например, `R2 = 0.90` означает, что модель объясняет около 90% вариативности `G3` на выбранной проверочной выборке.

## Результаты

Последний запуск `python -m src.run_training` сохранил таблицу:

```text
reports/tables/model_metrics.csv
```

Полученные метрики:

| model | features | MAE | MSE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dummy Regressor | with grades | 3.005924 | 14.357664 | 3.789151 | -0.001268 |
| Dummy Regressor | without grades | 3.050646 | 14.582650 | 3.818724 | -0.000079 |
| Linear Regression | with grades | 1.153341 | 2.185903 | 1.478480 | 0.847561 |
| Linear Regression | without grades | 2.620221 | 10.976946 | 3.313147 | 0.247200 |
| Gradient Boosting | with grades | 0.845542 | 1.120127 | 1.058361 | 0.921885 |
| Gradient Boosting | without grades | 0.899979 | 1.269058 | 1.126525 | 0.912968 |

Основные выводы:

- `DummyRegressor` почти не объясняет данные: `R2` около нуля.
- `LinearRegression` хорошо работает при наличии `G1/G2`, но заметно слабее без них.
- `Gradient Boosting` показывает лучшее качество в обоих сценариях.
- Сценарий без `G1/G2` остается сильным для бустинга: `MAE = 0.899979`, `R2 = 0.912968`.
- Расширенный датасет содержит много повторов, поэтому используется `GroupShuffleSplit`, чтобы одинаковые строки признаков не попадали одновременно в train и test.

## Графики

Сейчас сохраняются три графика:

```text
reports/figures/r2_comparison.png
reports/figures/mae_comparison.png
reports/figures/actual_vs_predicted_gb_with_grades.png
```

`r2_comparison.png` показывает сравнение моделей по `R2`.

`mae_comparison.png` показывает среднюю абсолютную ошибку в баллах.

`actual_vs_predicted_gb_with_grades.png` показывает соответствие фактических и предсказанных значений для градиентного бустинга с `G1/G2`.

## Воспроизводимость

- Все случайные разбиения и модели используют `random_state=42`, где это применимо.
- CSV-файлы не хранятся в Git.
- Метрики и выводы добавляются только после реального запуска кода.
- Для расширенного датасета используется групповое разбиение через `GroupShuffleSplit`.

## Возможные Улучшения

- Сохранить параметры бустинга в таблицу метрик вместе с результатами эксперимента.
- Добавить `n_iter` в итоговую таблицу, чтобы видеть, сколько итераций реально прошел бустинг.
- Построить график факт/прогноз для модели без `G1/G2`.
- Добавить график остатков для лучшей модели.
- Добавить анализ важности признаков в `src`.
- Сделать отдельный модуль с текстовыми выводами по метрикам и графикам.
- Аккуратно сравнить несколько наборов параметров градиентного бустинга.
- Добавить финальную проверку запуска проекта из чистого окружения.

Дальнейшие обновления будут обдуманны командой SLADKYU ink. Ждём и ваши предложения тоже.

