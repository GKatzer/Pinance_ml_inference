import lightgbm as lgb, numpy as np, pandas as pd

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
y = 3 * X["a"] - X["c"]
booster = lgb.Booster(model_str=lgb.train({"verbose": -1}, lgb.Dataset(X, y), 20).model_to_string())

row = X.iloc[[0]]
print("booster.feature_name():", booster.feature_name())
print("own names      :", booster.predict(row)[0])
print("renamed column :", booster.predict(row.rename(columns={"c": "renamed"}))[0])
print("order swapped  :", booster.predict(row[["a", "c", "b"]])[0])
