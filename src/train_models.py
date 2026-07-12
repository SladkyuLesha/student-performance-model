from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def train_dummy_regressor(X_train, y_train) -> DummyRegressor:
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_model(
    X_train,
    y_train,
    max_iter: int = 10000,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 40,
    l2_regularization: float = 1.0,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 30,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model
