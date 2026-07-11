# src/check_preprocessing.py

from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import preprocess_data, make_feature_sets
from src.splitting import honest_split_data


def main():
    data_path = Path("data/raw/student_data_500k.csv")

    df = load_data(data_path)
    df_prepared = preprocess_data(df)

    X_with_grades, X_without_grades, y = make_feature_sets(df_prepared)

    X_train_with_grades, X_test_with_grades, y_train_with, y_test_with = honest_split_data(X_with_grades, y)
    X_train_without_grades, X_test_without_grades, y_train_without, y_test_without = honest_split_data(X_without_grades, y)

    print(f"df_head:\n{df.head()}")
    print("Raw data:", df.shape)
    print("Prepared data:", df_prepared.shape)
    print("X with grades:", X_with_grades.shape)
    print("X without grades:", X_without_grades.shape)
    print("Target:", y.shape)
    print("Train set:", X_train_with_grades.shape, y_train_with.shape)
    print("Test set:", X_test_with_grades.shape, y_test_with.shape)
    print("Train set (without grades):", X_train_without_grades.shape, y_train_without.shape)
    print("Test set (without grades):", X_test_without_grades.shape, y_test_without.shape)


if __name__ == "__main__":
    main()