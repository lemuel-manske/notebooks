import copy
import math
import pandas as pd


R = pd.read_csv("data.csv", header=None)

print(R.describe())
# Qual a média de preço das casas?
# > 340412.7
# Quanto custa a menor casa?
# > 169900.00
# Quantos quartos tem a casa mais cara?
# > 5

R = R.to_numpy()
R = R.tolist()


RESULTS_COL_IDX = 2

X_1_COL_IDX = 0
X_2_COL_IDX = 1


def get_data():
    r = []

    for i in range(len(R)):
        r.append([])
        r[i].insert(0, 1)
        r[i].insert(1, R[i][X_1_COL_IDX])
        r[i].insert(2, R[i][X_2_COL_IDX])

    return r


def get_expected_results():
    r = []

    for i in range(len(R)):
        r.append([])
        r[i].append(R[i][RESULTS_COL_IDX])

    return r


def transposed(arr): # OK, 20:23
    r = []

    for _ in range(len(arr[0])): # len=3
        r.append([])

    for i in range(len(arr)): # len=3
        for j in range(len(arr[i])): # len=3
            r[j].insert(i, arr[i][j])

    return r


def identity(arr): # OK, 20:01
    n = len(arr)

    r = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                r[i][j] = 1

    return r


def upper_t(arr): # OK, 20:58
    new_arr = copy.deepcopy(arr)
    ident = identity(arr)

    n = len(new_arr)

    for i in range(n):
        p = new_arr[i][i]
        next_idxs = [k for k in range(n) if k > i]

        if p == 0:
            raise Exception("Não é possível calcular a inversa de uma matriz singular.")

        for k in next_idxs:
            x = new_arr[k][i]
            m = x / p

            for l in range(n):
                new_arr[k][l] = new_arr[k][l] - m * new_arr[i][l]
                ident[k][l] = ident[k][l] - m * ident[i][l]

    return new_arr, ident


def backward_subs(arr): # OK, 21:32
    t_s, i_t_s = upper_t(arr)

    n = len(t_s)

    r = copy.deepcopy(t_s)
    i_r = copy.deepcopy(i_t_s)

    for i in range(n - 1, -1, -1):
        p = r[i][i]
        next_idxs = [k for k in range(n) if k < i]

        if p == 0:
            raise Exception("Não é possível calcular a inversa de uma matriz singular.")

        for k in next_idxs:
            x = r[k][i]
            m = x / p

            for l in range(n):
                r[k][l] = r[k][l] - m * r[i][l]
                i_r[k][l] = i_r[k][l] - m * i_r[i][l]

    return r, i_r


def normalize(arr): # OK, 21:37
    r_s, i_r_s = backward_subs(arr)

    n = len(r_s)

    r_n = copy.deepcopy(r_s)
    i_n_s = copy.deepcopy(i_r_s)

    for i in range(n):
        pivot = r_n[i][i]

        for j in range(n):
            r_n[i][j] = r_n[i][j] / pivot
            i_n_s[i][j] = i_n_s[i][j] / pivot

    return r_n, i_n_s


def determinant(arr): # OK, 20:18
    n = len(arr)

    if n == 1:
        return arr[0][0]

    if n == 2:
        diag_p = []

        for i in range(n):
            diag_p.append(arr[i][i])

        diag_s = []

        for i in range(n):
            for j in range(n):
                if i + j == n - 1:
                    diag_s.append(arr[i][j])

        result_p = 1
        for x in diag_p:
            result_p *= x

        result_s = 1
        for x in diag_s:
            result_s *= x

        return result_p - result_s

    t_s, _ = upper_t(arr)

    r = 1
    for i in range(n):
        r = r * t_s[i][i]

    return r


def inversed(arr): # OK, 21:37
    return normalize(arr)[1]


def mul(arr_a, arr_b):
    rows_a = len(arr_a)
    cols_a = len(arr_a[0])

    rows_b = len(arr_b)
    cols_b = len(arr_b[0])

    # A(m x n) * B(n x p)
    if cols_a != rows_b:
        raise Exception('não consigo multiplicar')

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for m in range(rows_a):
        for p in range(cols_b):
            for n in range(cols_a):
                result[m][p] += arr_a[m][n] * arr_b[n][p]

    return result


def B(X, y):
    x_T = transposed(X)
    c = mul(x_T, X)
    c_inversa = inversed(c)
    r = mul(c_inversa, x_T)
    return mul(r, y)


def run_multiple_regression(X, y):
    return mul(X, B(X, y))


def avg(arr):
    return sum(arr) / len(arr)


def corr(X, y) -> float:
    avg_x = avg(X)
    avg_y = avg(y)

    de: float = 0.0
    dv1: float = 0.0
    dv2: float = 0.0

    for x, y in zip(X, y):
        de += (x - avg_x) * (y - avg_y)
        dv1 += (x - avg_x)**2
        dv2 += (y - avg_y)**2

    dv = math.sqrt(dv1 * dv2)

    r = de/dv

    return r


def lin_reg(x_, X, y):
    avg_x = avg(X)
    avg_y = avg(y)

    b1: float = 0.0

    de: float = 0.0
    dv: float = 0.0

    for x, y in zip(X, y):
        de += (x - avg_x) * (y - avg_y)
        dv += (x - avg_x)**2

    b1 = de / dv

    b0 = avg_y - (b1 * avg_x)

    y = b0 + (b1 * x_)

    return b0, b1, y


def run_demo():
    import matplotlib.pyplot as plt

    X = get_data()
    y = get_expected_results()

    house_sizes = [x[1] for x in X]
    house_prices = [y[0] for y in y]
    house_bedrooms = [x[2] for x in X]

    print("Correlação Tamanho casa x Preço casa:")
    print(corr(house_sizes, house_prices))

    print("Regressão linear Tamanho casa x Preço casa:")
    b0, b1, _ = lin_reg(2000, house_sizes, house_prices)
    print(f"b0: {b0}, b1: {b1}")

    plt.scatter(house_sizes, house_prices, label="y")
    plt.plot(house_sizes, [b0 + b1 * x for x in house_sizes], color='red', label="Linha de regressão")

    plt.xlabel("Tamanho da casa (sq ft)")
    plt.ylabel("Preço da casa (R$)")

    plt.title("Tamanho da casa vs Preço da casa")
    plt.legend()

    plt.show()

    print("Correlação Quantidade quartos x Preço casa:")
    print(corr(house_bedrooms, house_prices))

    print("Regressão linear Quantidade quartos x Preço casa:")
    b0, b1, _ = lin_reg(3, house_bedrooms, house_prices)
    print(f"b0: {b0}, b1: {b1}")

    plt.scatter(house_bedrooms, house_prices, label="y")

    plt.plot(house_bedrooms, [b0 + b1 * x for x in house_bedrooms], color='red', label="Linha de regressão")

    plt.xlabel("Quantidade de quartos")
    plt.ylabel("Preço da casa (R$)")

    plt.title("Quantidade de quartos vs Preço da casa")
    plt.legend()

    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    b = B(X, y)

    house_sizes = np.array([x[1] for x in X])
    house_bedrooms = np.array([x[2] for x in X])
    house_prices = np.array([value[0] for value in y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        house_sizes,
        house_bedrooms,
        house_prices,
        label="Dados observados",
    )

    size_grid, bedroom_grid = np.meshgrid(
        np.linspace(house_sizes.min(), house_sizes.max(), 30),
        np.linspace(house_bedrooms.min(), house_bedrooms.max(), 30),
    )

    price_grid = (
        b[0][0]
        + b[1][0] * size_grid
        + b[2][0] * bedroom_grid
    )

    ax.plot_surface(
        size_grid,
        bedroom_grid,
        price_grid,
        alpha=0.6,
    ) # Plano de regressão

    ax.set_xlabel("Tamanho da casa (sq ft)")
    ax.set_ylabel("Quantidade de quartos")
    ax.set_zlabel("Preço da casa (R$)")
    ax.set_title(
        "Tamanho da casa e quantidade de quartos vs. preço da casa"
    )

    plt.show()


if __name__ == "__main__":
    run_demo()
