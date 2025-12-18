import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

array_1 = np.array([2, 1, 3, 0, 233])

array_mean = np.mean(array_1) # mean is very sensitive to outliers

array_median = np.median(array_1) # median is robust to outliers, because it evaluates values based on their rank, not their size

array_variance = np.var(array_1) # finds the average of the squared differences between data-points and their mean

array_std = np.std(array_1) # standard deviation is the square-root of the variance


print(f'''
Orginal Data: {array_1}

Mean (Skewed by outlier): {array_mean:.2f}

Median (robust to outlier): {array_median}

Variance: {array_variance:.2f}

Standard Deviation: {array_std:.3f}
''')


# ======================================== Matplotlib & Seaborn ========================================


figure, axes = plt.subplots(figsize = (10, 6))

axes.plot(array_1, label = "Numpy Array", marker = "o", linestyle = "-") # notice we used axes.plot, not plt.plot (we plot on the axes, not the figure)

# Visualizing Mean vs. Median

axes.axhline(array_mean, c = "r", linestyle = "--", label = f"Mean ({array_mean:.2f})")

axes.axhline(array_median, c = "green", linestyle = "-", label = f"Median ({array_median:.2f})")

axes.set_title("Array, Mean vs. Median")

axes.set_xlabel("X-Axis")

axes.set_ylabel("Y-Axis")

axes.legend()

print(f"Figure Object {type(figure)}")

print(f"Axes Object {type(axes)}")

plt.show()










































































"""
We will demonstrate the Object-Oriented approach (Segment 1.2)
while enforcing the Hierarchy concepts (Segment 1.1).
"""

print("\n===================================== Visualizing Data (The Architect Way) =====================================\n")

# let's create fake data to work with
t_x = np.linspace(0, 10, 100)

t_y1 = np.sin(t_x)

t_y2 = np.cos(t_x)

"""
Matplotlib Hierachy:
1. The figure: This is the wall, or picture stand we're working on. It's job is to hold every drawing you create
2. The Axes: This is the canvas we place on the figure. You can have 1 or multiple canvases
"""

figure, (axes1, axes2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 6))

print(f'''
Type of "figure": {type(figure)}

Type of "axes1": {type(axes1)}
''')

# ===================================== Plotting on canvas 1 (axes1) =====================================

# we use the axes to plot, not plt
# not plt.plot(), but axes1.plot()

axes1.plot(t_x, t_y1, c = "b", label = "Sine Wave")

axes1.set_title("The Sine Wave (Canvas 1)") # Note: it's .set_title(), not .title()

axes1.set_ylabel("Amplitude (h)")
axes1.legend(loc = "upper left")
axes1.grid(True, alpha = 0.3)

# ===================================== Plotting on Canvas 2 (axes2) =====================================

axes2.plot(t_x, t_y2, c = "r", linestyle = "--", label = "Cosine Wave")

# setting titles and labels...
axes2.set_title("The Cosine Wave (Canvas 2)")
axes2.set_xlabel("Time (s)")
axes2.set_ylabel("Amplitude")


axes2.legend(loc = "upper right")

axes2.grid(True, alpha = 0.4)

# we use .tight_layout() to adjust the layout so the Wall doesn't look crowded

plt.tight_layout()

print("Plot generated successfully using OO approach.")
plt.show()









































































np.random.seed(199)

# generating sample data with noise using np.random
x_axis = np.linspace(0, 10, 100)

y_axis = np.sin(x_axis) + np.random.normal(0, 0.1, 100) # we're creating a sine wave and adding a bit of noise



# ===================================== LOW RESOLUTION =====================================

# extracting figure and axes
low_res_figure, low_res_axes = plt.subplots(
    figsize = (8, 4),

    dpi = 80
)


# Plotting on low resolution axes
low_res_axes.plot(x_axis, y_axis, c = "orange", linestyle = '--')

# setting title and labels for x & y axis...
low_res_axes.set_title("Web Preview (Low DPI)", fontsize = 13)

low_res_axes.set_xlabel("Time (s)")

low_res_axes.set_ylabel("Amplitude")



# ===================================== HIGH RESOLUTION =====================================

high_res_figure, high_res_axes = plt.subplots(figsize = (10, 7), dpi = 150)

high_res_axes.plot(x_axis, y_axis, c = 'crimson', linewidth = 2, linestyle = '--')


high_res_axes.set_title("Print Quality (High DPI)")

high_res_axes.set_xlabel("Time (s)")

high_res_axes.set_ylabel("Amplitude")

# ===================================== SAVING LOW RES IMAGE =====================================

# saving low_res_figure as low resolution png Raster image...
low_res_figure.savefig("sine_wave_low_pic.png", bbox_inches = "tight")


# ===================================== SAVING HIGH RES IMAGE =====================================

# saving high_res_figure as high resolution svg Vector image...
high_res_figure.savefig("sine_wave_high_pic.svg", format = "svg", bbox_inches = "tight")


# saving high_res_figure as high resolution pdf Vector...
high_res_figure.savefig("sine_wave_high_pic.pdf", format = "pdf", bbox_inches = "tight")

plt.show()

# cleaning up memory (very Important in large scale production)
plt.close(low_res_figure)
plt.close(high_res_figure)













































































entries = 300

server_logs = pd.DataFrame({
    "latency_ms": np.random.normal(loc = 150, scale = 30, size = entries), # Normal dist
    "throughput_mb": np.random.uniform(low = 10, high = 100, size = entries), # Uniform dist
    "server_region": np.random.choice(['US-East', 'EU-West'], size = entries), # Categorical
    "status": np.random.choice(['Healthy', 'Warning'], size = entries, p = [0.8, 0.2]) # Categorical
})



# ===================================== AXES-LEVEL APPROACH =====================================

# creating canvas

figure, (axes_1, axes_2, axes_3) = plt.subplots(1, 3, figsize = (16, 5))

# Plotting a Scatterplot on axes_1
sns.scatterplot(data = server_logs, x = "throughput_mb", y = "latency_ms", hue = 'status', ax = axes_1)

# setting the title for axes_1...
axes_1.set_title("Network Latency vs. Throughput")


# setting the content for axes_2
sns.boxplot(
    data = server_logs,
    x = 'server_region',
    y = 'latency_ms',
    ax = axes_2
)

# setting the title for axes_2 and axes_3...
axes_2.set_title("Latency Distribution by Region")
axes_3.set_title("Testing on axes_3 with Scatterplot of sample")



# setting the content for axes_3
axes_3.scatter(
    server_logs["throughput_mb"].sample(25),
    server_logs["latency_ms"].sample(25),    
    c = 'orange'
)


figure.suptitle("Axes-Level Functions",
    fontsize = 19,
    color = 'red',
    fontweight = 'bold'
)

figure.tight_layout()

plt.close(figure)




# ===================================== FIGURE-LEVEL APPROACH =====================================

fig_level = sns.relplot(
    data = server_logs,
    x = 'throughput_mb',
    y = 'latency_ms',
    hue = 'status',
    col = 'server_region',
    kind = 'scatter',
    height = 4,
    aspect = 5
)

# plt.tight_layout()

fig_level.fig.suptitle('Latency Analysis Faceted by Region', y = 1.03)

plt.show()







