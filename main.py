# Code 1: Even/Odd Classifier
import torch
import torch.nn as nn

X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
Y = torch.tensor([0,1,0,1])

model = nn.Linear(1, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(400):
    output = model(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(torch.argmax(model(torch.tensor([[7.0]]))).item())

# Code 2: Pass/Fail Classification
import torch
import torch.nn as nn

X = torch.tensor([[40.0],[55.0],[75.0],[90.0]])
Y = torch.tensor([0,0,1,1])

model = nn.Linear(1, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(400):
    output = model(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(torch.argmax(model(torch.tensor([[65.0]]))).item())

# Code 3: Above-10 Classifier
import torch
import torch.nn as nn

X = torch.tensor([[5.0],[10.0],[15.0]])
Y = torch.tensor([0,0,1])

model = nn.Linear(1, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(500):
    output = model(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(torch.argmax(model(torch.tensor([[12.0]]))).item())
