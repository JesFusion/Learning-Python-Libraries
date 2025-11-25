import numpy as np


raw_data = np.arange(120)

raw_data_rs = raw_data.reshape(6, 20)

raw_data_ts = raw_data_rs.T


train, test = np.vsplit(raw_data_rs, 2)

print(f'''
Original Array (Shape: {raw_data.shape}):
{raw_data}


Reshaped Array (Shape: {raw_data_rs.shape}):
{raw_data_rs}


Tranposed Array (Shape: {raw_data_ts.shape}):
{raw_data_ts}


Train (Shape: {train.shape}):
{train}


Train (Shape: {test.shape}):
{test}
''')

