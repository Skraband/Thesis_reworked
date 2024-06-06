import pandas as pd

name = "res/experiments/05_20_2024_21_00_45__PWN-ReadPowerPKL.pkl"
unpickled_df = pd.read_pickle(name)
print(unpickled_df[3]['MSE'])

y=0
for x in unpickled_df[3]['MSE']:
    y += unpickled_df[3]['MSE'][x]

#x = y/4
y = y/4
print(y)