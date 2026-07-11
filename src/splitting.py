import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def honest_split_data(X: pd.DataFrame,
                      y: pd.Series,
                      test_size: float = 0.2,
                      random_state: int = 42) -> tuple[
                          pd.DataFrame,
                          pd.DataFrame,
                          pd.Series,
                          pd.Series]:
    
    groups = pd.util.hash_pandas_object(X, index=False)

    splitter = GroupShuffleSplit(n_splits=1,
                                 test_size=test_size,
                                 random_state=random_state)

    train_pos, test_pos = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_pos]
    X_test = X.iloc[test_pos]
    y_train = y.iloc[train_pos]
    y_test = y.iloc[test_pos]

    return X_train, X_test, y_train, y_test