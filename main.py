from linear import Linear
from tensor import Tensor
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sgd import SGD
from mseloss import MSELoss

class CustomDataset():
    def __init__(self, file_path, selected_columns=None, label_column=None):
        self.data = pd.read_csv(file_path)  # 读取CSV文件
        self.selected_columns = selected_columns
        self.label_column = label_column
        if(self.selected_columns == None):
            self.selected_columns = self.data.columns
        if(self.label_column == None):
            self.label_column = self.selected_columns[-1]
        
        self.data_full = self.data.copy()  # 保存完整的数据
        self.Handling_missing_values()
    
    def Handling_missing_values(self):
        self.data = self.data.dropna()

    def print(self):
        dataset = self.data[self.selected_columns]
        print(dataset)

    def discretization(self):
        # print("-"*50)
        for column in self.selected_columns:
            data_column = self.data[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                # print(f'Column [{column}] need to be discretized (No Number Relationship! If needed, please code separately)')
                self.data[column] = self.data[column].astype('category').cat.codes
            data_column = self.data_full[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                # print(f'Column [{column}] need to be discretized (No Number Relationship! If needed, please code separately)')
                self.data_full[column] = self.data_full[column].astype('category').cat.codes
    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean)/std
        mean = self.data_full[legal_columns].mean()
        std = self.data_full[legal_columns].std()
        self.data_full[legal_columns] = (self.data_full[legal_columns] - mean)/std
    
    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        return self.data[legal_columns].to_numpy()
    def gety(self):
        return self.data[self.label_column].to_numpy()
    def __len__(self):
        return len(self.data)
    

dataset = CustomDataset('data/iris.csv', selected_columns=["sepal.length","sepal.width","petal.length","petal.width","species"])
dataset.discretization()
dataset.normalized()

X = dataset.getx()
y = dataset.gety()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = Tensor(X_train, dtype="float32")
X_test = Tensor(X_test, dtype="float32")
y_train = Tensor(y_train, dtype="long")
y_test = Tensor(y_test, dtype="long")

model = Linear(in_features=4, out_features=1)

def train(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions = model.forward(X_train)

        # Compute loss
        loss = criterion(predictions, y_train)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.data}')

def predict(model, X):
    return model.forward(X)

optimizer = SGD(params=[model.weight, model.bias], lr=0.01)
criterion = MSELoss()

# Train the model
train(model, criterion, optimizer, X_train, y_train, epochs=100)