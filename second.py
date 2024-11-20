import numpy as np
import pandas as pd
import scipy.special as special
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры
k1 = 5
k2 = 10
n = 90

# Генерация выборки из F-распределения
sample_f = np.random.f(dfnum=k1, dfden=k2, size=n)

# Гистограмма относительных частот и теоретическая кривая распределения
x = np.linspace(min(sample_f), max(sample_f), 1000)
pdf_f = stats.f.pdf(x, dfn=k1, dfd=k2)

plt.figure(figsize=(12, 6))
plt.hist(sample_f, bins=num_bins, edgecolor='black', density=True, alpha=0.6, label='Относительные частоты')
plt.plot(x, pdf_f, 'r-', label='Теоретическая кривая распределения')
plt.title('Гистограмма относительных частот и теоретическая кривая распределения для F-распределения')
plt.xlabel('Интервалы')
plt.ylabel('Относительные частоты')
plt.legend()
plt.show()

# Бокс-плот распределения
plt.figure(figsize=(12, 6))
plt.boxplot(sample_f, vert=False)
plt.title('Бокс-плот распределения для F-распределения')
plt.xlabel('Значения')
plt.show()

# Статистическая интерпретация бокс-плота
outliers_f = np.sum((sample_f < np.percentile(sample_f, 25) - 1.5 * stats.iqr(sample_f)) |
                    (sample_f > np.percentile(sample_f, 75) + 1.5 * stats.iqr(sample_f)))
print(f'Теоретически ожидаемое число выбросов для F-распределения: {outliers_f}')