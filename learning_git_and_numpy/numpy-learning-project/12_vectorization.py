import numpy as np
import time

'''
Vectorization is what makes numpy lists faster than traditional python lists. Here's why:

1. The computer performs the operation on every element at once
2. It is written in C, which is blazing fast compared to python
3. All the element data types are the same, so there's no need for type checking, which saves time and makes processing faster
'''

# ============================= testing addition of corresponding elements in manual python lists =============================

f_list = list(range(1_000_000))

s_list = list(range(1_000_000))

result_l = []

# get start time, loop and get end time
start_time = time.time()

for x in range(len(f_list)):

    result_l.append(f_list[x] + s_list[x])

end_time = time.time()

python_duration = end_time - start_time

print(f'''
First Five results: {result_l[:5]}


Python Loop Time: {python_duration:.6f} seconds
''')


# ============================= trying the same operation with Numpy =============================


array_1 = np.arange(1_000_000)

array_2 = np.arange(1_000_000)


N_start_time = time.time()

N_result = array_1 + array_2

N_end_time = time.time()

numpy_duration = N_end_time - N_start_time


print(f'''
Numpy Vectorization Time: {numpy_duration:.6f}


Python Time is {python_duration:.6f} while Numpy time is {numpy_duration:.6f}
''')

# Vectorization applies to all math operations, not just addition

an_array = np.arange(4)

print(f'''
Original Array: {an_array}

Array multiplied by 3: {an_array * 3}

Sine of Array: {np.sin(an_array)}
''')