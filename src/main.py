import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

SAMPLE_COUNT = 2500
SAMPLE_SIZE = 2500


# нормальное распределение
MU, SIGMA = 0, 1
rv = stats.norm(loc=MU, scale=SIGMA)
# генерация выборок
samples = rv.rvs(size=(SAMPLE_COUNT, SAMPLE_SIZE))

# 1. асимптотическая нормальность среднего, дисперсии, медианы
# статистики
means = np.mean(samples, axis=1)  # среднее
variances = np.var(samples, axis=1, ddof=1)  # дисперсия
medians = np.median(samples, axis=1)  # выборочный квантиль порядка 0.5 <=> медиана


def plot_histogram(
    data: np.ndarray, title: str, nrows: int, ncols: int, index: int
) -> None:
    plt.subplot(nrows, ncols, index)
    plt.hist(data, bins=50, density=True, alpha=0.6, color="b", label="Гистограмма")
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), "r-", label="Норм. распр.")
    plt.title(title)
    plt.legend()


plot_histogram(means, "Выборочное среднее", 1, 3, 1)
plot_histogram(variances, "Выборочная дисперсия", 1, 3, 2)
plot_histogram(medians, "Выборочная медиана", 1, 3, 3)
plt.tight_layout()
plt.show()


# 2. проверка сходимости nF(X(2)) -> U1 ~ Г(2, 1)
F = rv.cdf  # функция распределения
X_sorted = np.sort(samples, axis=1)  # type: ignore
X2 = X_sorted[:, 1]  # вторая порядковая статистика (X(2))
Xn = X_sorted[:, -1]  # последняя порядковая статистика (X(n))

U1 = SAMPLE_COUNT * F(X2)  # n * F(X(2))
U2 = SAMPLE_COUNT * (1 - F(Xn))  # n * (1 - F(X(n)))


x1 = np.linspace(0, np.max(U1), 100)
x2 = np.linspace(0, np.max(U2), 100)

plt.subplot(1, 2, 1)
plt.hist(U1, bins=50, density=True, alpha=0.6, color="b", label="Гистограмма")
plt.plot(x1, stats.gamma.pdf(x1, a=2, scale=1), "r-", label="Г(2,1)")
plt.legend()
plt.title("nF(X_{(2)})")

plt.subplot(1, 2, 2)
plt.hist(U2, bins=50, density=True, alpha=0.6, color="b", label="Гистограмма")
plt.plot(x2, stats.expon.pdf(x2), "r-", label="Exp(1)")
plt.legend()
plt.title("n(1 - F(X_{(n)}))")

plt.tight_layout()
plt.show()
