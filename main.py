from linear import Linear
from tensor import Tensor
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["species"]).values
y = LabelEncoder().fit_transform(df["species"])
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = Tensor(X_train, dtype="float32")
X_test = Tensor(X_test, dtype="float32")
y_train = Tensor(y_train, dtype="long")
y_test = Tensor(y_test, dtype="long")

model = Linear(in_features=4, out_features=1)