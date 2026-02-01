import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

np.random.seed(42)

# параметры задачи

P = 4e6
Vmin = 0.4
rho = 7800
sigma_allow = 9.5e6

#границы переменных проектирования
bounds = np.array([
    [0.2, 1.0], #r
    [0.2, 1.8], #l
    [0.01, 0.15], #ts
    [0.01, 0.1]  # th
])

#целевая функция (МАССА, которую будет аппроксимировать GP)

def mass_function(x):
    R, L, Ts, Th = x
    #объем материала
    Vmat = (4/3)*np.pi*((R+Th)**3 - R**3) + np.pi*L*((R+Ts)**2 - R**2)
    #масса
    mass = rho * Vmat
    return mass


#функция штрафов

def penalty_function(x):
    R, L, Ts, Th = x

    #объем сосуда
    V = (4/3)*np.pi*R**3 + np.pi*R**2*L

    #напряжение
    sigma = max(P*R/(2*Th), P*R/Ts)

    penalty = 0

    if V < Vmin:
        penalty += 1e7

    if sigma > sigma_allow:
        penalty += 1e7

    return penalty


#полная целевая функция (mass + penalty)

def objective(x):
    return mass_function(x) + penalty_function(x)


#expected improvement(критерий выбора новых точек), те функция выбирает где лучше всего попробовать новую точку

def expected_improvement(X, model, best_y, eps=1e-12):
    mu, sigma = model.predict(X, return_std=True) #или предсказание или неопределенность 
    sigma = np.maximum(sigma, eps)
    Z = (best_y - mu) / sigma #на сколько лучше текущего минимума
    EI = (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    EI[sigma == 0.0] = 0.0 #если неопределенность нулевая улучшения нет
    return EI.ravel() #возвращаем плоский массив


#случайные точки

def random_points(n):
    X = np.zeros((n, 4))
    for i in range(4):
        X[:, i] = np.random.uniform(bounds[i, 0],
                                    bounds[i, 1],
                                    size=n) #генерируем равномерно в пределах bounds
    return X


#bayesian optimization

#стартовые точки 

X = random_points(15)

#ВАЖНО: gaussian process обучается ТОЛЬКО НА МАССЕ

Y = np.array([mass_function(x) for x in X]) 


#gaussian process модель

kernel = Matern(nu=2.5)

model = GaussianProcessRegressor(kernel=kernel,
                                 alpha=1e-6,
                                 normalize_y=True)


#40 итераций байесовского поиска

for i in range(40): 

    #обучение surrogate модели
    model.fit(X, Y)

    #кандидаты
    X_test = random_points(3000)

    best_y = np.min(Y) #текущее лучшее решение (по массе)

    EI = expected_improvement(X_test, model, best_y) #cчитаем полезность каждой точки

    #лучшая точка
    x_next = X_test[np.argmax(EI)]

    #вычисляем настоящую массу
    y_next = mass_function(x_next)

    #добавляем данные
    X = np.vstack((X, x_next))
    Y = np.append(Y, y_next)

    print(f"Iteration {i+1:02d} | Best mass = {np.min(Y):.2f} kg")


#результат (с учетом штрафов)

total_values = np.array([objective(x) for x in X])

best_index = np.argmin(total_values) #индекс минимальной целевой функции

print("\nFINAL RESULT")
print("R  =", round(X[best_index][0], 4))
print("L  =", round(X[best_index][1], 4))
print("Ts =", round(X[best_index][2], 4))
print("Th =", round(X[best_index][3], 4))

print("Mass =", round(mass_function(X[best_index]), 2), "kg")
print("Penalty =", round(penalty_function(X[best_index]), 2))
print("Objective =", round(total_values[best_index], 2))
