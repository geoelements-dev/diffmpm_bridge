import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the results
results_df = pd.read_csv("loss_results.csv")

# Extract data for plotting
param1 = results_df['loc']
param2 = results_df['std']
param3 = results_df['height']
loss = results_df['loss']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(param1, param2, param3, c=loss, cmap='viridis')
ax.set_xlabel('loc')
ax.set_ylabel('std')
ax.set_zlabel('height')
ax.set_yscale('log')
fig.colorbar(scatter, label='Loss')
plt.show()