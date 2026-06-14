import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go


# setting
FILE_NAME = "data.xlsx"
FEATURE_COLUMNS = ["f1","feature2"]  # set a value ['feature1', 'feature2']
TARGET_COLUMN = "target"
NUM_EPOCHS = 2000			   # count step edication
LEARNING_RATE = 0.01			# speed edication (step gradient)


# 1. loading and preparing tensors
try:
	df = pd.read_excel(FILE_NAME)
except Exception as e:
	print(f"File read error: {e}")
	exit()

# translate data from Pandas in tenzors PyTorch (type float32)
X_train = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
y_train = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32).view(-1, 1)

# 2. creating model "underground"
# weight initialized with random values.
# the number of weights equals the number of feature columns.
W = torch.randn(X_train.shape[1], 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print("Initial random weights:", W.detach().numpy().flatten())
print("Inistial bias (b):", b.item())

# 3. training cycle (Gradient descent)
for epoch in range(NUM_EPOCHS):
	# forward pass: calculate the prediction using the formula Y = X*W + b
	y_pred = X_train @ W + b
	
	# calculating the MSE loss function manually using tensors
	loss = torch.mean((y_pred - y_train) ** 2)
	
	# backward pass: PyTorch calculate derivatives
	loss.backward()
	
	# Update weights (Gradient descent step)
	with torch.no_grad():
		W -= LEARNING_RATE * W.grad
		b -= LEARNING_RATE * b.grad
		
		# zero out the gradients before the next step
		W.grad.zero_()
		b.grad.zero_()
		
	if (epoch + 1) % 400 == 0:
		print(f"epoch {epoch+1}/{NUM_EPOCHS} | current MSE Loss: {loss.item():.4f}")

# finel parameters and error
final_mse = loss.item()
W_final = W.detach().numpy().flatten()
b_final = b.item()

print("\n--- update PyTorch compleated ---")
print(f"final Intercept (b): {b_final:.4f}")
for col, weight in zip(FEATURE_COLUMNS, W_final):
	print(f"final weight for {col}: {weight:.4f}")
print(f"MSE: {final_mse:.4f}")


# 4. visiolization result
# function for obtaining predictions from trained tensors
def predict_torch(X_np):
	X_tensor = torch.tensor(X_np, dtype=torch.float32)
	with torch.no_grad():
		return (X_tensor @ W + b).numpy()

if len(FEATURE_COLUMNS) == 1:
	x_min, x_max = X_train.numpy().min() - 1, X_train.numpy().max() + 1
	x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
	y_line = predict_torch(x_range)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=X_train.numpy().flatten(), y=y_train.numpy().flatten(), mode="markers", marker=dict(size=10, color="red"), name="Data"))
	fig.add_trace(go.Scatter(x=x_range.ravel(), y=y_line.ravel(), mode="lines", line=dict(color="blue", width=2), name=f"Линия PyTorch (MSE: {final_mse:.2f})"))
	fig.update_layout(title=f"PyTorch line regression | MSE = {final_mse:.3f}", xaxis_title=FEATURE_COLUMNS[0], yaxis_title=TARGET_COLUMN)
	fig.show()

elif len(FEATURE_COLUMNS) == 2:
	x_min, x_max = X_train[:, 0].numpy().min() - 1, X_train[:, 0].numpy().max() + 1
	y_min, y_max = X_train[:, 1].numpy().min() - 1, X_train[:, 1].numpy().max() + 1

	x_range = np.linspace(x_min, x_max, 20)
	y_range = np.linspace(y_min, y_max, 20)
	x_mesh, y_mesh = np.meshgrid(x_range, y_range)

	grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
	z_mesh = predict_torch(grid_points).reshape(x_mesh.shape)

	fig = go.Figure()
	fig.add_trace(go.Scatter3d(x=X_train[:, 0].numpy(), y=X_train[:, 1].numpy(), z=y_train.numpy().flatten(), mode="markers", marker=dict(size=5, color="red"), name="Data"))
	fig.add_trace(go.Surface(x=x_range, y=y_range, z=z_mesh, colorscale="Blues", opacity=0.6, name="Plane PyTorch", showscale=False))
	fig.update_layout(title=f"PyTorch regression (3D) | MSE = {final_mse:.3f}", scene=dict(xaxis_title=FEATURE_COLUMNS[0], yaxis_title=FEATURE_COLUMNS[1], zaxis_title=TARGET_COLUMN))
	fig.show()