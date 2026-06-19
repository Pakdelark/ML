import os 
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go


# setting
FILE_NAME = "data.xlsx"
FEATURE_COLUMNS = ["f1","feature2"]  # set a value ['feature1', 'feature2']
TARGET_COLUMN = "target"
NUM_EPOCHS = 10000			   # count step edication
LEARNING_RATE = 0.1			# speed edication (step gradient)
POLY_DEGREE = 3	 # polinomial degree: 1 - linear, 2 - quadratic, 3 - cubic
MODEL_FILE = "model_weights.pth"
TEST_FILE_NAME = "test_data.xlsx" # data for test

# mode work 
# True -> traning model and save to file
# False -> uploading ready model from file 
TRAIN_MODE = True



# 1. loading and preparing tensors
try:
	df = pd.read_excel(FILE_NAME)
except Exception as e:
	print(f"File read error: {e}")
	exit()

X_raw = df[FEATURE_COLUMNS].values
y_raw = df[TARGET_COLUMN].values.reshape(-1, 1)

# manual data scaling (MinMax) 
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)

# protection against "division by zero" if max = min
X_range_diff = np.where((X_max - X_min) == 0, 1, X_max - X_min)

# scale x_raw to to the range [0, 1]
X_scaled = (X_raw - X_min) / X_range_diff

# function for manual polynomial feature generation
def generate_poly_features(X_np, degree):
	X_tensor = torch.tensor(X_np, dtype=torch.float32)
	poly_list = []
	
	# for the degree 1 and higher adding basic signs and their degrees
	for d in range(1, degree + 1):
		poly_list.append(X_tensor ** d)
	
	# if parametrs 2 and degree >= 2, adding their products (X1 * X2)
	if X_np.shape[1] == 2 and degree >= 2:
		x1 = X_tensor[:, 0:1]
		x2 = X_tensor[:, 1:2]
		poly_list.append(x1 * x2)
		if degree >= 3:
			poly_list.append((x1 ** 2) * x2)
			poly_list.append(x1 * (x2 ** 2))
			
	return torch.cat(poly_list, dim=1)

# generate polinomial signs for edication
if TRAIN_MODE:
	X_train = generate_poly_features(X_scaled, POLY_DEGREE)
	y_train = torch.tensor(y_raw, dtype=torch.float32)

	# calculate the average value of y for the formula R^2
	y_mean = torch.mean(y_train)

	# 2. creating model "underground"
	# weight initialized with random values.
	# the number of weights equals the number of feature columns.
	W = torch.randn(X_train.shape[1], 1, requires_grad=True)
	b = torch.randn(1, requires_grad=True)

	print(f"Initial random weights count: {W.shape[0]}")
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
			
		if (epoch + 1) % 2000 == 0:  # print epoch every 5000
			# сalculation of R² for the current epoch 
			with torch.no_grad():
				ss_res = torch.sum((y_train - y_pred) ** 2)
				ss_tot = torch.sum((y_train - y_mean) ** 2)
				r2_current = 1 - (ss_res / ss_tot) # 1 = ideal
			
			print(f"epoch {epoch+1}/{NUM_EPOCHS} | current MSE Loss: {loss.item():.3f} | RMSE {(loss.item())**0.5:.3f} | R^2 {r2_current.item():.3f}")

		# calculate the final R^2 after completing the training
		final_mse = loss.item()
		with torch.no_grad():
			ss_res = torch.sum((y_train - y_pred) ** 2)
			ss_tot = torch.sum((y_train - y_mean) ** 2)
			final_r2_val = (1 - (ss_res / ss_tot)).item()

	# saving everything to disk (weights, bias and axis scales)
	torch.save({
		'W': W.detach(),
		'b': b.detach(),
		'X_min': X_min,
		'X_max': X_max,
		'X_range_diff': X_range_diff,
		'r2': r2_current,
		'mse': loss.item(),
		'columns': FEATURE_COLUMNS,
		'degree': POLY_DEGREE
	}, MODEL_FILE)
	print(f"The model has been successfully saved to file:{MODEL_FILE}")
else:
	print("Mode: INFERENCE (loading the finished model)")
	if not os.path.exists(MODEL_FILE):	# searches for a file in the directory
		print(f"Wrong: file {MODEL_FILE} not found! Firs run the script with setting TRAIN_MODE = True")
		exit()

	# loading model memory from the file
	checkpoint = torch.load(MODEL_FILE, weights_only=False)
	W = checkpoint["W"]
	b = checkpoint['b']
	X_min = checkpoint['X_min']
	X_max = checkpoint['X_max']
	X_range_diff = checkpoint['X_range_diff']
	final_r2_val = checkpoint['r2']
	final_mse = checkpoint['mse']
	print("The model successfully loaded! Training cycle skipped")
	print("---------------------------------")

	# retrieve saved settings (with protection against old save files)
	saved_columns = checkpoint.get('columns', None)
	saved_degree = checkpoint.get('degree', None)

	# verification of the number of weights against current settings
	if saved_columns != FEATURE_COLUMNS or saved_degree != POLY_DEGREE:
		print(f"[ERROR] the saved model {W.shape[0]}, does not match {generate_poly_features(X_scaled, POLY_DEGREE).shape[1]}")
		if saved_columns != FEATURE_COLUMNS:
			print(f"  • columns in the file: {saved_columns} | current setting: {FEATURE_COLUMNS}")
		if saved_degree != POLY_DEGREE:
			print(f"  • polinomial degree in the file: {saved_degree} | current setting: {POLY_DEGREE}")
		print("---------------------------------")
		print("please set TRAIN_MODE = True to retrain the model with the new parameters")
		exit()

# finel parameters and error
W_final = W.detach().numpy().flatten()
b_final = b.item()

print("\n--- update PyTorch completed ---")
print(f"final Intercept (b): {b_final:.3f}")
print(f"final weights: {np.round(W_final, 3)}")
print(f"Train MSE: {final_mse:.3f} | Train RMSE {final_mse**0.5:.3f} | Train R^2: {final_r2_val:.3f}")



# function for obtaining predictions from trained tensors
def predict_poly_torch(X_np):
	X_np_scaled = (X_np - X_min) / X_range_diff  # scaling the test grid 
	X_poly = generate_poly_features(X_np_scaled, POLY_DEGREE)
	with torch.no_grad():
		return (X_poly @ W + b).numpy()

# calculate the prediction for the manual test set X_TEST
X_test_np, y_test_true, y_test_pred = None, None, None
if os.path.exists(TEST_FILE_NAME):
	try:
		df_test = pd.read_excel(TEST_FILE_NAME)
		X_test_np = df_test[FEATURE_COLUMNS].values
		y_test_pred = predict_poly_torch(X_test_np)
		print("\n--- Test file loaded successfully ---")

		# if the test file contains a column with the actual answers, we calculate deviation metrics
		if TARGET_COLUMN in df_test.columns:
			y_test_true = df_test[TARGET_COLUMN].values.reshape(-1, 1)

			# Count test MSE and R^2 via tenzors 
			y_test_true_t = torch.tensor(y_test_true, dtype=torch.float32)
			y_test_pred_t = torch.tensor(y_test_pred, dtype=torch.float32)
			
			test_mse = torch.mean((y_test_pred_t - y_test_true_t) ** 2).item()
			
			ss_res = torch.sum((y_test_true_t - y_test_pred_t) ** 2)
			ss_tot = torch.sum((y_test_true_t - torch.mean(y_test_true_t)) ** 2)
			test_r2 = (1 - (ss_res / ss_tot)).item() if ss_tot != 0 else 0.0
			print(f"Test MSE (Отклонение): {test_mse:.3f} | Test RMSE: {test_mse**0.5:.3f} | Test R^2: {test_r2:.3f}")
		else:
			print("Note: Target column missing in test file. Calculating predictions only.")
			for i, x_t in enumerate(X_test_np):
				print(f"Point {i+1} | X = {x_t} -> Predicted Y = {y_test_pred[i][0]:.3f}")
	except Exception as e:
		print(f"Warning: Failed to process test file: {e}. Skipping test stage.")
		X_test_np = None
else:
	print(f"\nNote: Test file '{TEST_FILE_NAME}' not found. Visualizing training data only.")

# 4. visiolization result
if len(FEATURE_COLUMNS) == 1:
	x_grid = np.linspace(X_raw.min() - 1, X_raw.max() + 1, 300).reshape(-1, 1)
	y_line = predict_poly_torch(x_grid)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=X_raw.flatten(), y=y_raw.flatten(), mode="markers", marker=dict(size=10, color="red"), name="Excel data"))
	fig.add_trace(go.Scatter(x=x_grid.ravel(), y=y_line.ravel(), mode="lines", line=dict(color="blue", width=2.5), name=f"Poly Line (R^2: {final_r2_val:.2f})"))
	if X_test_np is not None:
		fig.add_trace(go.Scatter(x=X_test_np.flatten(), y=y_test_pred.flatten(), mode="markers", marker=dict(size=10, color="green", symbol="diamond"), name="test prediction"))
	fig.update_layout(title=f"PyTorch Polynomial Regression (Degree {POLY_DEGREE}) | MSE = {final_mse:.3f} | R^2 = {final_r2_val:.3f}", xaxis_title=FEATURE_COLUMNS[0], yaxis_title=TARGET_COLUMN)
	fig.show()

elif len(FEATURE_COLUMNS) == 2:
	x_min_val, x_max_val = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
	y_min_val, y_max_val = X_raw[:, 1].min() - 1, X_raw[:, 1].max() + 1

	x_range = np.linspace(x_min_val, x_max_val, 30)
	y_range = np.linspace(y_min_val, y_max_val, 30)
	x_mesh, y_mesh = np.meshgrid(x_range, y_range)

	grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
	z_mesh = predict_poly_torch(grid_points).reshape(x_mesh.shape)

	fig = go.Figure()
	fig.add_trace(go.Scatter3d(x=X_raw[:, 0], y=X_raw[:, 1], z=y_raw.flatten(), mode="markers", marker=dict(size=5, color="red"), name="Exel data"))
	fig.add_trace(go.Surface(x=x_range, y=y_range, z=z_mesh, colorscale="Blues", opacity=0.6, name="Plane PyTorch", showscale=False))
	if X_test_np is not None:
		fig.add_trace(go.Scatter3d(x=X_test_np[:, 0], y=X_test_np[:, 1], z=y_test_pred.flatten(), mode="markers", marker=dict(size=5, color="green", symbol="diamond"), name="test prediction"))
	fig.update_layout(title=f"PyTorch Poly Regression (3D Degree {POLY_DEGREE}) | MSE = {final_mse:.3f} | RMSE {(loss.item())**0.5:.3f} | R^2 = {final_r2_val:.3f}", scene=dict(xaxis_title=FEATURE_COLUMNS[0], yaxis_title=FEATURE_COLUMNS[1], zaxis_title=TARGET_COLUMN))
	fig.show()