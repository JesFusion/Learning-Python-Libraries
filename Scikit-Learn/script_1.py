import numpy as np
np.random.seed(19)


dataset = np.random.randint(34, 77, size = (3, 2))

dataset[:, -1] = dataset[:, -1] * 1093

print(f'''
======================================== Original Dataset (Shape: {dataset.shape}) ========================================

{dataset}
''')

# ===================================== axis = 0 =====================================

# in axis = 0, we collapse rows into one row per column

average_values = np.mean(dataset, axis = 0)

std_values = np.std(dataset, axis = 0)


print(f'''
===================================== Statistics per Feature =====================================
      
Mean (Age, Salary): {average_values}

Standard Deviation (Age, Salary): {average_values}
''')


