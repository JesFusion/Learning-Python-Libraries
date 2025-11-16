import numpy as np
np.random.seed(19)


# let's create a vector and a matrix...

matrix_A = np.random.randint(34, 77, size = (5, 4))

vector_B = np.linspace(34, 77, 5, dtype = np.int32)


print(f'''
Maxtix A [Shape: {matrix_A.shape}]:
{matrix_A}


Vector B [Shape: {vector_B.shape}]: {vector_B}
''')


# broadcasting the vector will fail
# (5, 4) + (5, ) = (5, 4) + (, 5)
# analyzing from right to left:
# 4 vs. 5  = INCOMPATIBLE!!
# this operation CANNOT be performed and will result to a ValueError

try:

    matrix_A + vector_B

except ValueError as e:

    print(f'I told you this would result to a ValueError: "{e}"')



# the solution to this is to make the vector "vector_B" a column vector using np.newaxis

col_vector_B = vector_B[:, np.newaxis]


print(f'''
Original vector_B:
{vector_B}

Original shape: {vector_B.shape}

New vector_B:
{col_vector_B}

New shape (using np.newaxis): {col_vector_B.shape}
''')


# let's try broadcasting again
# it should work this time...

# (5, 4) + (5, 1)

# analyzing from right to left:
# 4 vs. 1 = COMPATIBLE!! (Rule 2)
# 5 vs. 5 = COMPATIBLE!! (Rule 2)


try:
    print(f'''
    Result of matrix_A and new vector_B:
{matrix_A + col_vector_B}
    ''')

except ValueError as bummer:

    print(f"Damn! It was supposed to work. Try checking the error out:\n{bummer}")
