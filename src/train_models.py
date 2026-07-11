from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

def train_dummy_regressor(X_train, y_train) -> DummyRegressor:
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    return dummy_model


def train_linear_regression(X_train, y_train):
    linear_model_group = LinearRegression()
    linear_model_group.fit(X_train, y_train)
    return linear_model_group


def train_gradient_boosting_model(X_train, y_train):
    gradient_boosting_model = HistGradientBoostingRegressor(
        max_iter=10000,
        early_stopping=True,
        random_state=42
        )
    gradient_boosting_model.fit(X_train, y_train)
    return gradient_boosting_model