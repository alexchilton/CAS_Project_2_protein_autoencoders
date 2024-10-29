import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have the distance matrix
distance_matrix = np.random.rand(118, 7)  # Example distance matrix

# Convert the distance matrix to a DataFrame
distance_df = pd.DataFrame(distance_matrix)

# Example: Plotting the heatmap (for visualization)
sns.heatmap(distance_matrix, cmap="viridis", square=True)
plt.show()

# Now `distance_df` contains the data from the heatmap
print(distance_df)

# Assuming you have a heatmap plot
ax = sns.heatmap(distance_matrix, cmap="viridis", square=True)
plt.show()

# Extract the data from the heatmap
heatmap_data = ax.collections[0].get_array().reshape(distance_matrix.shape)

# Convert the extracted data to a DataFrame
extracted_df = pd.DataFrame(heatmap_data)
print(extracted_df)