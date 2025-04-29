import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

file_path = "C:\Users\HowardLimHF\OneDrive - Asia Pacific University\Documents\GitHub\Predictive-Modeling-of-Tuberculosis-Incidence-Rates\Tuberculosis_Trends.csv"
df = pd.read_csv(file_path)

df = df.drop('Country', axis=1)

df = pd.get_dummies(df, columns=['Region', 'Income_Level'])

scaler = MinMaxScaler()
scaled_columns = df.columns
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

X = df.drop('TB_Incidence_Rate', axis=1)
y = df['TB_Incidence_Rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

class TuberculosisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TuberculosisDataset(X_train_tensor, y_train_tensor)
test_dataset = TuberculosisDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TBIncidenceModel(nn.Module):
    def __init__(self, input_dim):
        super(TBIncidenceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1) 
        self.dropout = nn.Dropout(0.3) 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_dim = X_train.shape[1]
model = TBIncidenceModel(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

num_epochs = 100
best_val_loss = float('inf')
best_model_state = None

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(inputs)  
        
        loss = criterion(outputs.squeeze(), labels)  
        loss.backward() 
        optimizer.step()  
        
        running_loss += loss.item()
    

    scheduler.step()
    
    model.eval()  
    with torch.no_grad():
        val_loss = criterion(model(X_test_tensor).squeeze(), y_test_tensor)
    
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss.item())
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = model.state_dict()

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss.item()}")

model.load_state_dict(best_model_state)

model.eval() 
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    test_loss = criterion(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item()}")

    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2}")

# Plot training vs. validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot predicted vs actual TB incidence rates with regression line
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual TB Incidence Rate')
plt.ylabel('Predicted TB Incidence Rate')
plt.title('Actual vs Predicted TB Incidence Rate')
plt.legend()
plt.show()
