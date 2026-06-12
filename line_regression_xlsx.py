import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 0. ------- settings --------
file_name = "/home/xrix/Documents/py_progect/ML/ML_edication/data.xlsx"


# name of collums-parametrs (x) and target colums (Y)
feature_columns = ["feature1", "feature2"]  # and more parametrs
target_column = "target"  # predicting

# 1. ------- uploading data ------
try:
	df = pd.read_excel(file_name)
	print("Data loaded")
except Exception as e:
	print(f'Error reading file: {e}')
	exit()

# selecting features and the target variable
X = df[feature_columns].values
y = df[target_column].values

# 2. ---- training a line regression model ---- 
model = LinearRegression()
model.fit(X,y)

# exit coefficients of the equation
print("\n---result of training---")
print(f"Intercept (bias): {model.intercept_:.4f}")  # value at zero
for col, coef in zip(feature_columns, model.coef_):
	print(f"coefficients for {col}: {coef:.4f}")

# 3. ---- visualization (work for 2-parametrs) ---- 
if len(feature_columns) == 2:
	print("\nBilding interactive chart")

	# creating a grid for constructing the regression plane
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	x_range = np.linspace(x_min, x_max, 20)
	y_range = np.linspace(y_min, y_max, 20)
	x_mesh, y_mesh = np.meshgrid(x_range, y_range)

	# predict z-values (plane) for each grid point
	grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
	z_mesh = model.predict(grid_points).reshape(x_mesh.shape)

	# creating a 3D plot using Plotly
	fig = go.Figure()

	# adding real points from Excel
	fig.add_trace(
		go.Scatter3d(
			x=X[:,0],
			y=X[:,1],
			z=y,
			mode="markers",
			marker=dict(size=5, color="red", opacity=0.8),
			name="real data",
		)
	)
	# adding a regression plane
	fig.add_trace(
		go.Surface(
			x=x_range,
			y=y_range,
			z=z_mesh,
			colorscale="Blues",
			opacity=0.6,
			name="Плоскость регрессии",
			showscale=False,
		)
	)

	# axis settings matching your column names
	fig.update_layout(
		title="Linear Regression: Prediction Plane",
		scene=dict(
			xaxis_title=feature_columns[0],
			yaxis_title=feature_columns[1],
			zaxis_title=target_column,
		),
		margin=dict(l=0, r=0, b=0, t=40),
	)

	# opens the chart in the browserе
	fig.show()

else:
	print(
		f"\nThe model has been successfully trained for {len(feature_columns)} parameters."
	)
	print(
		"3D visualization is available for exactly two independent parameters."
	)




