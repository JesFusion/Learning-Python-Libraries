import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data: 10 server ping times (in ms)
# We add one massive outlier (1000ms) to see how it breaks the mean
server_latency = np.array([20, 22, 19, 24, 21, 23, 18, 20, 200, 25])

# 2. Mean (Average)
# Sensitive to outliers. Notice how the '200' drags this number up.
mean_latency = np.mean(server_latency)

# 3. Median (Mid-point)
# Robust to outliers. This ignores the '200' and gives the "real" typical value.
median_latency = np.median(server_latency)

# 4. Variance (Var)
# The average squared distance from the mean. Hard to interpret directly because units are squared (ms^2).
variance_latency = np.var(server_latency)

# 5. Standard Deviation (Std)
# The square root of Variance. Returns to original units (ms).
# In a Normal Distribution, 68% of data falls within Mean +/- 1 Std.
std_latency = np.std(server_latency)

print(f"Data: {server_latency}")
print(f"Mean (skewed by outlier): {mean_latency:.2f}")   # Likely around 39.2
print(f"Median (true center): {median_latency:.2f}")     # Likely around 21.5
print(f"Standard Deviation: {std_latency:.2f}")










# 1. Create the Objects explicitly (The OO Method)
# fig is the window, ax is the drawing area.
fig, ax = plt.subplots(figsize=(10, 6)) # [cite: 13]

# 2. Plotting data onto the Axes object
# Notice we use ax.plot, NOT plt.plot
# We plot the raw server latency data
ax.plot(server_latency, label='Server Latency', marker='o', linestyle='-')

# 3. Integrating NumPy Stats: Visualizing Mean vs Median
# This visually proves why Mean is dangerous with outliers
ax.axhline(mean_latency, color='red', linestyle='--', label=f'Mean ({mean_latency:.1f})')
ax.axhline(median_latency, color='green', linestyle='-', label=f'Median ({median_latency:.1f})')

# 4. Modifying the Attributes (Setters)
# In OO, we often use 'set_' prefixes
ax.set_title("Server Latency: Mean vs Median Sensitivity")
ax.set_xlabel("Request ID")
ax.set_ylabel("Latency (ms)")

# 5. Adding the legend (it reads the 'label' arguments from above)
ax.legend()

# 6. Show the hierarchy structure
print(f"Figure Object: {type(fig)}")
print(f"Axes Object: {type(ax)}")

plt.show()