import os 
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from itertools import combinations_with_replacement
''' 
future add: 
- regularization L1 and L2
- adaptive POLY_DEGREE
- display of training parameters at test values
'''

# setting
FILE_NAME = "data.xlsx"				# main traning dataset 
TEST_FILE_NAME = "test_data.xlsx"	# data for test
FEATURE_COLUMNS = []				# read from the file data.xlsx
TARGET_COLUMN = None				# read last column from the file data.xlsx
X_TEST = None						# read from the file test_data
NUM_EPOCHS = 10000					# count step edication
LEARNING_RATE = 0.1					# speed edication (step gradient)
EARLY_STOP_TOLERANCE = 1e-7			# stopping criterion: if the weight changes by less than this value, training stops
POLY_DEGREE = 4						# polinomial degree: 1 - linear, 2 - quadratic, 3 - cubic
MODEL_FILE = "model_weights.pth"


# mode work 
# True -> traning model and save to file
# False -> uploading ready model from file 
TRAIN_MODE = True

# 1. loading and preparing tensors
try:
	df = pd.read_excel(FILE_NAME)
	# remove system index columns often created by Excel/Pandas
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	# remove any rows containing empty cells (protection against NoneType and NaN)
	df = df.dropna()
except Exception as e:
	print(f"[CRITICAL ERROR] failed to process the main file: {e}")
	exit()
if df.empty or df.shape[1] < 2:
	print("[CRITICAL ERROR] the table must contain at least two columns (column and target) and must not be empty!")
	exit()

# adaptive definition: The last column is always the target; everything else consists of features.
TARGET_COLUMN = df.columns[-1]
FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]
print(f"Features for training automatically detected: {FEATURE_COLUMNS} -> [{TARGET_COLUMN}]")

# check of the test file
df_test = None
if os.path.exists(TEST_FILE_NAME):
	try:
		df_test = pd.read_excel(TEST_FILE_NAME)
		df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')].dropna()
		
		missing_cols = [col for col in FEATURE_COLUMNS if col not in df_test.columns]
		if missing_cols:
			print(f"[CRITICAL ERROR] test file is missing the required feature columns: {missing_cols}")
			exit()
		print("--- test file passed validation successfully ---")
	except Exception as e:
		print(f"[CRITICAL ERROR] test file validation error: {e}")
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
	num_samples, num_features = X_np.shape
	X_tensor = torch.tensor(X_np, dtype=torch.float32)
	poly_list = []
	
	# Automatic combine feature for every degree
	for d in range(1, degree + 1):
		for combos in combinations_with_replacement(range(num_features), d):
			current_feature = torch.prod(X_tensor[:, combos], dim=1, keepdim=True)
			poly_list.append(current_feature)
	
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
	print(f"Initial bias (b): {b.item():.3f}")

# 3. training cycle (Gradient descent)
	for epoch in range(NUM_EPOCHS):
		if epoch > 0:
			W_old = W.clone().detach()
			b_old = b.clone().detach()
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

		# check the stopping criterion starting from the second epoch
		if epoch > 0:
			with torch.no_grad():
				# calculate the maximum absolute change among the weights and the bias
				w_change = torch.max(torch.abs(W - W_old)).item()
				b_change = torch.abs(b - b_old).item()
				
				if w_change < EARLY_STOP_TOLERANCE and b_change < EARLY_STOP_TOLERANCE:
					print(f"\n[Early Stopping] training has been halted at the epoch {epoch+1}")
					print(f"The weights stabilized W: {w_change:.2e}, b: {b_change:.2e}")
					break

		if (epoch + 1) % 2000 == 0:  # print epoch every 2000
			# сalculation of R² for the current epoch 
			with torch.no_grad():
				ss_res = torch.sum((y_train - y_pred) ** 2)
				ss_tot = torch.sum((y_train - y_mean) ** 2)
				r2_current = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0 # 1 = ideal
			
			print(f"epoch {epoch+1}/{NUM_EPOCHS} | current MSE Loss: {loss.item():.3f} | RMSE {(loss.item())**0.5:.3f} | R^2 {r2_current.item():.3f}")

		# calculate the final R^2 after completing the training
		final_mse = loss.item()
		with torch.no_grad():
			ss_res = torch.sum((y_train - y_pred) ** 2)
			ss_tot = torch.sum((y_train - y_mean) ** 2)
			final_r2_val = (1 - (ss_res / ss_tot)).item() if ss_tot != 0 else 0.0

	# saving everything to disk (weights, bias and axis scales)
	torch.save({
		'W': W.detach(),
		'b': b.detach(),
		'X_min': X_min,
		'X_max': X_max,
		'X_range_diff': X_range_diff,
		'r2': final_r2_val,
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

print(f"final Intercept (b): {b_final:.3f}")
print(f"final weights: {np.round(W_final, 3)}")
print(f"Train MSE: {final_mse:.3f} | Train RMSE {final_mse**0.5:.3f} | Train R^2: {final_r2_val:.3f}")
print("---------------------------------")


# function for obtaining predictions from trained tensors
def predict_poly_torch(X_np):
	X_np = np.array(X_np).reshape(-1, len(FEATURE_COLUMNS))
	X_np_scaled = (X_np - X_min) / X_range_diff  # scaling the test grid 
	X_poly = generate_poly_features(X_np_scaled, POLY_DEGREE)
	with torch.no_grad():
		return (X_poly @ W + b).numpy()

# calculate the prediction for the manual test set X_TEST
X_test_np, y_test_true, y_test_pred = None, None, None
test_mse, test_r2 = None, None
if df_test is not None:
	X_test_np = df_test[FEATURE_COLUMNS].values
	y_test_pred = predict_poly_torch(X_test_np)

	# if the test file contains a column with the actual answers, we calculate deviation metrics
	if TARGET_COLUMN in df_test.columns:
		y_test_true = df_test[TARGET_COLUMN].values.reshape(-1, 1)

		# Count test MSE and R^2 via tenzors 
		y_test_true_t = torch.tensor(y_test_true, dtype=torch.float32)
		y_test_pred_t = torch.tensor(y_test_pred, dtype=torch.float32)
		
		# calculate the true deviation of predictions from the test of initial responses.
		test_mse = torch.mean((y_test_pred_t - y_test_true_t) ** 2).item()
		
		ss_res = torch.sum((y_test_true_t - y_test_pred_t) ** 2)
		y_test_true_mean = torch.mean(y_test_true_t)
		ss_tot = torch.sum((y_test_true_t - y_test_true_mean) ** 2)
		
		test_r2 = (1 - (ss_res / ss_tot)).item() if ss_tot != 0 else 0.0
		print(f"Test MSE (aberration): {test_mse:.3f} | Test RMSE: {test_mse**0.5:.3f} | Test R^2: {test_r2:.3f}")
	else:
		print("Note: Target column missing in test file. Calculating predictions only.")
		for i, x_t in enumerate(X_test_np):
			print(f"Point {i+1} | X = {x_t} -> Predicted Y = {y_test_pred[i][0]:.3f}")
else:
	print(f"\nNote: Test file '{TEST_FILE_NAME}' not found. Visualizing training data only.")

# generating titles for graphs
title_text = f"PyTorch Poly Regression (Degree {POLY_DEGREE})<br>TRAIN: MSE = {final_mse:.3f} | R^2 = {final_r2_val:.3f}"
if test_mse is not None and test_r2 is not None:
    title_text += f"<br>TEST:   MSE = {test_mse:.3f} | R^2 = {test_r2:.3f}"

# 4. visiolization result
if len(FEATURE_COLUMNS) == 1:
	x_grid = np.linspace(X_raw.min() - 1, X_raw.max() + 1, 300).reshape(-1, 1)
	y_line = predict_poly_torch(x_grid)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=X_raw.flatten(), y=y_raw.flatten(), mode="markers", marker=dict(size=10, color="red"), name="Excel data"))
	fig.add_trace(go.Scatter(x=x_grid.ravel(), y=y_line.ravel(), mode="lines", line=dict(color="blue", width=2.5), name=f"Poly Line (R^2: {final_r2_val:.2f})"))
	if X_test_np is not None:
		x_display = X_test_np[:, 0]
		y_display = y_test_true.flatten() if y_test_true is not None else y_test_pred.flatten()
		name_display = "test real data" if y_test_true is not None else "test prediction"
		fig.add_trace(go.Scatter(x=x_display.flatten(), y=y_display.flatten(), mode="markers", marker=dict(size=10, color="green", symbol="diamond"), name=name_display))
	fig.update_layout(title=title_text, xaxis_title=FEATURE_COLUMNS[0], yaxis_title=TARGET_COLUMN)
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
	fig.add_trace(go.Scatter3d(x=X_raw[:, 0], y=X_raw[:, 1], z=y_raw.flatten(), mode="markers", marker=dict(size=6, color="red"), name="Exel data"))
	fig.add_trace(go.Surface(
		x=x_range, 
		y=y_range, 
		z=z_mesh, 
		colorscale="Blues", 
		opacity=0.6, 
		name="Plane PyTorch", 
		showscale=False
	))
	if X_test_np is not None:
		# if the test set contains an answer column, we use it for the Z-axis; otherwise, we use the predictions.
		if y_test_true is not None:
			z_display = y_test_true.flatten()
			name_display = "test real data"
		else:
			z_display = y_test_pred.flatten()
			name_display = "test model predictions"
			
		fig.add_trace(go.Scatter3d(
			x=X_test_np[:, 0], 
			y=X_test_np[:, 1], 
			z=z_display, 
			mode="markers", 
			marker=dict(size=6, color="green", symbol="diamond"), 
			name=name_display
		))
	fig.update_layout(title=title_text, scene=dict(xaxis_title=FEATURE_COLUMNS[0], yaxis_title=FEATURE_COLUMNS[1], zaxis_title=TARGET_COLUMN))
	fig.show()