# see https://www.youtube.com/watch?v=OGpQxIkR4ao
# 1) Design Model
# 2) Consider Loss and Optimizer
# 3) Training Loop
# - forward pass
# - backward pass
# - update weights1
#

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target


n_samples, n_features = X.shape
print(n_samples, n_features)
# 569, 30

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# scle it
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Concert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Design Model
# f = wx + b, sigmoid at the end
#


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(
            n_input_features, 1
        )  # the 1 is that one variable is predicted

    def forward(self, x):

        y_predicted = torch.sigmoid(self.linear(x))

        return y_predicted


model = LogisticRegression(n_features)


# 2) Consider Loss and Optimizer

criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 100


for epoch in range(n_epochs):

    # foward pass and loss calculation
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update weihts
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch={epoch + 1}, loss={loss.item():.4f}")

    # print(f"epoch={epoch + 1}, loss={loss.item():.4f}")


# Evaluation
# evalutation should not be part of the computational graph where we want to track the history
# this is why the context manager. Otherwise this would go into the gradient calcuations
with torch.no_grad():

    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()

    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")


# loss .2396
# accuracy in the video: .8947
