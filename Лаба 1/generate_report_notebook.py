"""Utility script to generate the lab report notebook."""

from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Лабораторная работа 1. Линейная регрессия и PCA\n\n"
            "В лабораторной работе изучаем процесс построения регрессионных моделей "
            "на примере датасета *diamonds*. Рассматриваем классическую линейную "
            "регрессию, гребневую регрессию и сокращение размерности признаков с "
            "помощью анализа главных компонент (PCA)."
        ),
        nbf.v4.new_markdown_cell(
            "## Введение\n"
            "Цель работы — освоить базовый цикл работы с моделями линейной регрессии: "
            "изучить исходные данные, выполнить предобработку, оценить мультиколлинеарность, "
            "сравнить качество моделей с и без регуляризации, а затем проверить, как метод "
            "главных компонент влияет на результаты. В работе применяются библиотеки "
            "`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib` и `seaborn`."
        ),
        nbf.v4.new_markdown_cell(
            "## Описание датасета\n"
            "Используется набор данных [diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds), "
            "который содержит характеристики бриллиантов (вес, качество огранки, цвет, чистота, физические "
            "размеры) и их цену. Всего 53 940 наблюдений. Целевая переменная — `price` (стоимость камня). "
            "Ниже показаны основные визуализации, подготовленные скриптом `text.py`:\n\n"
            "- распределения числовых признаков,\n"
            "- распределение целевой переменной `price`,\n"
            "- корреляционная матрица.\n\n"
            "![Numeric distributions](figures/numeric_feature_distributions.png)\n\n"
            "![Price distribution](figures/price_distribution.png)\n\n"
            "![Correlation matrix](figures/correlation_matrix.png)"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "DATA_PATH = Path('diamonds.csv')\n\n"
            "data = pd.read_csv(DATA_PATH).drop(columns=['Unnamed: 0'])\n"
            "data.head()"
        ),
        nbf.v4.new_code_cell("data.describe(include='all')"),
        nbf.v4.new_markdown_cell(
            "### Анализ пропусков\n"
            "В наборе отсутствуют пропущенные значения — таблица ниже показывает нули для всех столбцов."
        ),
        nbf.v4.new_code_cell("data.isna().sum()"),
        nbf.v4.new_markdown_cell(
            "## Подготовка данных\n"
            "Для дальнейшего моделирования выполняются следующие шаги:\n\n"
            "- **Разделение выборки.** Используется `train_test_split` c долей тестовой выборки 20 % и фиксированным `random_state=42`.\n"
            "- **Масштабирование числовых признаков.** Применяется `StandardScaler` внутри `ColumnTransformer`.\n"
            "- **Кодирование категориальных признаков.** Используется `OneHotEncoder` с отключением ошибок на неизвестных значениях.\n"
            "- **Мультиколлинеарность.** Рассчитаны коэффициенты VIF для числовых признаков; результаты сохранены в `results/vif_values.csv`."
        ),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LinearRegression, Ridge\n"
            "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error\n"
            "from sklearn.model_selection import KFold, cross_validate\n"
            "from sklearn.decomposition import PCA\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n\n"
            "RANDOM_STATE = 42\n"
            "TEST_SIZE = 0.2\n\n"
            "numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']\n"
            "categorical_features = ['cut', 'color', 'clarity']\n\n"
            "X = data.drop(columns=['price'])\n"
            "y = data['price']\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE\n"
            ")\n\n"
            "numeric_pipeline = Pipeline([('scaler', StandardScaler())])\n"
            "categorical_pipeline = Pipeline([\n"
            "    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n"
            "])\n\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', numeric_pipeline, numeric_features),\n"
            "    ('cat', categorical_pipeline, categorical_features)\n"
            "], remainder='drop', sparse_threshold=0)\n\n"
            "vif_df = pd.DataFrame({\n"
            "    'feature': numeric_features,\n"
            "    'vif': [variance_inflation_factor(\n"
            "        data[numeric_features].replace(0, 1e-6).values, i\n"
            "    ) for i in range(len(numeric_features))]\n"
            "})\n"
            "vif_df"
        ),
        nbf.v4.new_markdown_cell(
            "## Ход работы\n"
            "Сначала обучаем две модели на исходных признаках:\n\n"
            "1. Классическая линейная регрессия.\n"
            "2. Гребневая регрессия (`alpha = 10`).\n\n"
            "Обе модели оцениваются на тестовой выборке (RMSE, R², MAPE) и через 5-кратную кросс-валидацию. "
            "Результаты загружаем из `results/metrics_original_features.csv`."
        ),
        nbf.v4.new_code_cell(
            "metrics_original = pd.read_csv('results/metrics_original_features.csv')\n"
            "metrics_original"
        ),
        nbf.v4.new_markdown_cell(
            "Далее снижаем мультиколлинеарность с помощью PCA. Перед понижением размерности используем такие же шаги "
            "предобработки, затем подбираем число компонент, сохраняя не менее 95 % дисперсии. График метода локтя "
            "(накопленная объяснённая дисперсия) показан ниже.\n\n"
            "![PCA explained variance](figures/pca_explained_variance.png)\n\n"
            "На полученных главных компонентах повторяем обучение линейной и гребневой регрессии."
        ),
        nbf.v4.new_code_cell(
            "metrics_pca = pd.read_csv('results/metrics_pca_features.csv')\n"
            "metrics_pca"
        ),
        nbf.v4.new_markdown_cell(
            "### Сводное сравнение моделей\n"
            "Ниже представлена объединённая таблица метрик для моделей на исходных признаках и на PCA-компонентах."
        ),
        nbf.v4.new_code_cell(
            "metrics_comparison = pd.read_csv('results/metrics_comparison.csv')\n"
            "metrics_comparison"
        ),
        nbf.v4.new_markdown_cell(
            "## Заключение\n"
            "- Исходный датасет не содержит пропусков, а признаки товара сильно коррелируют между собой (особенно размеры и вес). VIF показывает высокую мультиколлинеарность для `carat`, `x`, `y`, `z`.\n"
            "- Линейная и гребневая регрессии на исходных данных демонстрируют близкое качество, при этом гребневая обеспечивает более устойчивые результаты на кросс-валидации.\n"
            "- После применения PCA (с сохранением 95 % дисперсии) качество моделей остаётся сопоставимым: небольшое снижение RMSE компенсируется более стабильными метриками и устранением мультиколлинеарности.\n"
            "- Метод PCA помогает упростить модель без заметной потери точности и облегчает интерпретацию за счёт устранения коррелированных признаков."
        ),
    ]

    output_path = Path("report.ipynb")
    nbf.write(nb, output_path)


if __name__ == "__main__":
    main()
