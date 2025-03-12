import numpy as np
import pandas as pd

df = pd.read_csv("mobile_phones.csv")


# 1.
print("1.1. Поддерживает 2 сим-карты: ", df["dual_sim"].value_counts()[1])
print("1.2. Поддерживает 3G: ", df["three_g"].value_counts()[1])
print("1.3. Наибольшее число ядер у процессора: ", df["n_cores"].max())
