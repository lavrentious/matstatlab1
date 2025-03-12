import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

df = pd.read_csv("mobile_phones.csv")


# 1
print("1.1. Поддерживает 2 сим-карты: ", df["dual_sim"].value_counts()[1])
print("1.2. Поддерживает 3G: ", df["three_g"].value_counts()[1])
print("1.3. Наибольшее число ядер у процессора: ", df["n_cores"].max())

# 2
# 2.1
battery_power = df["battery_power"]
mean = np.mean(battery_power)
variance = np.var(battery_power, ddof=1)
q2_5 = np.quantile(battery_power, 0.4)
print("2.2.1 Выборочное среднее: ", mean)
print("2.2.2 Выборочная дисперсия: ", variance)
print("2.2.3 Выборочная квантиль 2/5: ", q2_5)


# 2.2
x = np.sort(battery_power)
y = np.arange(len(x)) / float(len(x))
plt.plot(x, y)
plt.title("Эмпирическая функция распределения")
plt.xlabel("Емкость батареи")
plt.ylabel("F")
plt.show()

# 2.3
battery_power_wifi = df[df["wifi"] == 1]["battery_power"]
battery_power_no_wifi = df[df["wifi"] == 0]["battery_power"]

# 2.3.1 гистограмма
plt.figure(figsize=(10, 5))
sns.histplot(battery_power, bins=15, color="grey", label="Все телефоны")
sns.histplot(battery_power_wifi, bins=15, color="red", label="Есть Wi-Fi")
sns.histplot(battery_power_no_wifi, bins=15, color="blue", label="Нет Wi-Fi")
plt.xlabel("Емкость батареи")
plt.ylabel("Частота")
plt.show()

# 2.3.1 box plot

df["wifi_label"] = df["wifi"].map({0: "Нет Wi-Fi", 1: "Есть Wi-Fi"})
df["category"] = "Все телефоны"
df_wifi = df.copy()
df_wifi["category"] = df_wifi["wifi_label"]

plt.figure(figsize=(10, 5))
sns.boxplot(
    x="category",
    y="battery_power",
    data=pd.concat([df, df_wifi]),
    palette=["gray", "red", "blue"],
)
plt.xlabel("Категория")
plt.ylabel("Емкость батареи")
plt.show()
