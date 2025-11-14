import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main() -> int:
	print("=" * 60)
	print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ: ПРЕДСКАЗАНИЕ ЦЕНЫ НА ДОМА")
	print("=" * 60)
	
	# 1) Load data
	print("\n[ШАГ 1] Загрузка данных...")
	if not os.path.exists(TRAIN_PATH):
		print(f"[ERROR] File not found: {TRAIN_PATH}")
		print("Place train.csv into data/raw/ and retry.")
		return 1
	
	df = pd.read_csv(TRAIN_PATH)
	print(f"[ШАГ 1] Данные загружены: {df.shape[0]} строк, {df.shape[1]} колонок")

	if "SalePrice" not in df.columns:
		print("[ERROR] Column 'SalePrice' not found in train.csv")
		return 1
	print("[ШАГ 1] Проверка: колонка 'SalePrice' найдена ✓")

	# 2) Target and features
	print("\n[ШАГ 2] Разделение данных на признаки и целевую переменную...")
	y = df["SalePrice"]
	X = df.drop(columns=["SalePrice"])
	print(f"[ШАГ 2] Целевая переменная (y): {len(y)} значений")
	print(f"[ШАГ 2] Признаки (X): {X.shape[0]} строк, {X.shape[1]} колонок")

	# 3) Column types
	print("\n[ШАГ 3] Определение типов признаков...")
	num_cols = X.select_dtypes(include=["int64", "float64"]).columns
	cat_cols = X.select_dtypes(include=["object"]).columns
	print(f"[ШАГ 3] Числовых признаков: {len(num_cols)}")
	print(f"[ШАГ 3] Категориальных признаков: {len(cat_cols)}")

	# 4) Preprocessing
	print("\n[ШАГ 4] Создание пайплайна предобработки...")
	numeric_pipe = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median"))
	])
	print("[ШАГ 4] Пайплайн для числовых: заполнение пропусков медианой")

	categorical_pipe = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore"))
	])
	print("[ШАГ 4] Пайплайн для категориальных: заполнение пропусков + One-Hot Encoding")

	preprocess = ColumnTransformer(
		transformers=[
			("num", numeric_pipe, num_cols),
			("cat", categorical_pipe, cat_cols),
		]
	)
	print("[ШАГ 4] ColumnTransformer создан ✓")

	# 5) Model
	print("\n[ШАГ 5] Создание модели...")
	model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
	print("[ШАГ 5] Модель: RandomForest с 300 деревьями")

	pipe = Pipeline(steps=[
		("prep", preprocess),
		("model", model)
	])
	print("[ШАГ 5] Полный пайплайн создан: предобработка → модель ✓")

	# 6) Cross-Validation
	print("\n[ШАГ 6] Запуск кросс-валидации (5-fold)...")
	print("[ШАГ 6] Это может занять несколько минут...")
	scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
	rmse_mean = (-scores).mean()
	rmse_std = (-scores).std()
	print(f"[ШАГ 6] Кросс-валидация завершена!")
	print(f"[RESULT] CV RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")

	# 7) Hold-out validation
	print("\n[ШАГ 7] Hold-out валидация...")
	print("[ШАГ 7] Разделение на обучающую (80%) и валидационную (20%) выборки...")
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
	print(f"[ШАГ 7] Обучающая выборка: {X_train.shape[0]} строк")
	print(f"[ШАГ 7] Валидационная выборка: {X_valid.shape[0]} строк")
	
	print("[ШАГ 7] Обучение модели на обучающей выборке...")
	pipe.fit(X_train, y_train)
	print("[ШАГ 7] Модель обучена ✓")
	
	print("[ШАГ 7] Предсказание на валидационной выборке...")
	valid_pred = pipe.predict(X_valid)
	print(f"[ШАГ 7] Сделано {len(valid_pred)} предсказаний")
	
	holdout_rmse = mean_squared_error(y_valid, valid_pred) ** 0.5
	print(f"[RESULT] Hold-out RMSE: {holdout_rmse:.4f}")

	# 8) Save model pipeline
	print("\n[ШАГ 8] Сохранение обученной модели...")
	model_path = os.path.join(MODELS_DIR, "baseline_random_forest.pkl")
	joblib.dump(pipe, model_path)
	print(f"[INFO] Модель сохранена: {model_path} ✓")

	print("\n" + "=" * 60)
	print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
	print("=" * 60)
	
	return 0


if __name__ == "__main__":
	sys.exit(main())




