import numpy as np
import pandas as pd
import scipy.special as special
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры
a = 3
sigma_squared = 16
sigma = np.sqrt(sigma_squared)
n = 90

# Генерация выборки из нормального распределения
np.random.seed(42)  # Для воспроизводимости
sample = np.random.normal(loc=a, scale=sigma, size=n)

# Группировка данных в интервалы и нахождение абсолютных частот
num_bins = int(np.ceil(1 + 3.322 * np.log10(n)))  # Правило Стерджеса
freq, bins = np.histogram(sample, bins=num_bins)

# Сумма абсолютных частот
sum_absolute_freq = np.sum(freq)

# Построение диаграммы абсолютных частот
plt.figure(figsize=(12, 6))
plt.hist(sample, bins=num_bins, edgecolor='black')
plt.title('Гистограмма абсолютных частот')
plt.xlabel('Интервалы')
plt.ylabel('Абсолютные частоты')
plt.show()

# Относительные частоты
relative_freq = freq / n

# Сумма относительных частот
sum_relative_freq = np.sum(relative_freq)

# Построение диаграммы относительных частот
plt.figure(figsize=(12, 6))
plt.hist(sample, bins=num_bins, edgecolor='black', density=True)
plt.title('Гистограмма относительных частот')
plt.xlabel('Интервалы')
plt.ylabel('Относительные частоты')
plt.show()

# Теоретическая кривая распределения
x = np.linspace(min(sample), max(sample), 1000)
pdf = stats.norm.pdf(x, loc=a, scale=sigma)

# Гистограмма относительных частот и теоретическая кривая распределения
plt.figure(figsize=(12, 6))
plt.hist(sample, bins=num_bins, edgecolor='black', density=True, alpha=0.6, label='Относительные частоты')
plt.plot(x, pdf, 'r-', label='Теоретическая кривая распределения')
plt.title('Гистограмма относительных частот и теоретическая кривая распределения')
plt.xlabel('Интервалы')
plt.ylabel('Относительные частоты')
plt.legend()
plt.show()

# Гистограмма абсолютных частот и теоретическая частота
plt.figure(figsize=(12, 6))
plt.hist(sample, bins=num_bins, edgecolor='black', alpha=0.6, label='Абсолютные частоты')
plt.plot(x, pdf * n * (bins[1] - bins[0]), 'r-', label='Теоретическая частота')
plt.title('Гистограмма абсолютных частот и теоретическая частота')
plt.xlabel('Интервалы')
plt.ylabel('Частота')
plt.legend()
plt.show()

# Эмпирическая функция распределения и теоретическая функция распределения
ecdf = np.cumsum(relative_freq)
cdf = stats.norm.cdf(bins, loc=a, scale=sigma)

plt.figure(figsize=(12, 6))
plt.step(bins[1:], ecdf, where='post', label='Эмпирическая функция распределения')
plt.plot(bins, cdf, 'r-', label='Теоретическая функция распределения')
plt.title('Эмпирическая и теоретическая функции распределения')
plt.xlabel('Значения')
plt.ylabel('Функция распределения')
plt.legend()
plt.show()

# Бокс-плот распределения
plt.figure(figsize=(12, 6))
plt.boxplot(sample, vert=False)
plt.title('Бокс-плот распределения')
plt.xlabel('Значения')
plt.show()

# Статистическая интерпретация бокс-плота
outliers = np.sum((sample < np.percentile(sample, 25) - 1.5 * stats.iqr(sample)) |
                  (sample > np.percentile(sample, 75) + 1.5 * stats.iqr(sample)))
print(f'Теоретически ожидаемое число выбросов: {outliers}')

# Параметры
q = 1.3

# Вероятность P(|X - MX| < qσ) вручную через функцию Лапласа
def laplace_function(x):
    return (1.0 + special.erf(x / np.sqrt(2.0))) / 2.0

prob_manual = laplace_function(q) - laplace_function(-q)
print(f'Вероятность P(|X - MX| < {q}сигма) вручную: {prob_manual}')

# Вероятность P(|X - MX| < qσ) с использованием встроенных функций
prob_python = stats.norm.cdf(q) - stats.norm.cdf(-q)
print(f'Вероятность P(|X - MX| < {q}сигма) с использованием встроенных функций: {prob_python}')

# Оценка вероятности по выборке
mx = np.mean(sample)
sigma = np.std(sample)
prob_sample = np.sum(np.abs(sample - mx) < q * sigma) / n
print(f'Оценка вероятности по выборке: {prob_sample}')

# Увеличение объема выборки в 50 раз и повторение вычислений
n_large = 50 * n
sample_large = np.random.normal(loc=a, scale=sigma, size=n_large)

prob_sample_large = np.sum(np.abs(sample_large - a) < q * sigma) / n_large
print(f'Оценка вероятности по выборке при увеличенном объеме: {prob_sample_large}')

# Точечные оценки параметров распределения

# 4.1. Непосредственное применение формул
mean_estimate = np.mean(sample)
median_estimate = np.median(sample)
variance_estimate = np.var(sample, ddof=1)
std_dev_estimate = np.std(sample, ddof=1)
skewness_estimate = stats.skew(sample)
kurtosis_estimate = stats.kurtosis(sample)

print(f'Среднее: {mean_estimate}')
print(f'Медиана: {median_estimate}')
print(f'Дисперсия: {variance_estimate}')
print(f'Стандартное отклонение: {std_dev_estimate}')
print(f'Коэффициент асимметрии: {skewness_estimate}')
print(f'Эксцесс: {kurtosis_estimate}')

# 4.2. Применение встроенных функций
describe = stats.describe(sample)
print(f'Среднее (встроенная функция): {describe.mean}')
print(f'Дисперсия (встроенная функция): {describe.variance}')
print(f'Стандартное отклонение (встроенная функция): {np.sqrt(describe.variance)}')
print(f'Коэффициент асимметрии (встроенная функция): {describe.skewness}')
print(f'Эксцесс (встроенная функция): {describe.kurtosis}')

# Увеличение объема выборки в 50 раз и повторение вычислений
describe_large = stats.describe(sample_large)
print(f'Среднее (увеличенная выборка): {describe_large.mean}')
print(f'Дисперсия (увеличенная выборка): {describe_large.variance}')
print(f'Стандартное отклонение (увеличенная выборка): {np.sqrt(describe_large.variance)}')
print(f'Коэффициент асимметрии (увеличенная выборка): {describe_large.skewness}')
print(f'Эксцесс (увеличенная выборка): {describe_large.kurtosis}')


print("////////////////////////////////////////////")

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