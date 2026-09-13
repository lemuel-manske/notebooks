import math
import copy


def transpose(A):
    r = []

    for _ in range(len(A[0])):
        r.append([])

    for i in range(len(A)):
        for j in range(len(A[i])):
            r[j].insert(i, A[i][j])

    return r


def upper_t(A):
    new_A = copy.deepcopy(A)
    ident = identity(A)

    n = len(new_A)

    for i in range(n):
        p = new_A[i][i]
        next_idxs = [k for k in range(n) if k > i]

        if p == 0:
            raise Exception("Não é possível calcular a inversa de uma matriz singular.")

        for k in next_idxs:
            x = new_A[k][i]
            m = x / p

            for l in range(n):
                new_A[k][l] = new_A[k][l] - m * new_A[i][l]
                ident[k][l] = ident[k][l] - m * ident[i][l]

    return new_A, ident


def backward_subs(A):
    t_s, i_t_s = upper_t(A)

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


def normalize(A):
    r_s, i_r_s = backward_subs(A)

    n = len(r_s)

    r_n = copy.deepcopy(r_s)
    i_n_s = copy.deepcopy(i_r_s)

    for i in range(n):
        pivot = r_n[i][i]

        for j in range(n):
            r_n[i][j] = r_n[i][j] / pivot
            i_n_s[i][j] = i_n_s[i][j] / pivot

    return r_n, i_n_s


def determinant(A):
    n = len(A)

    if n == 1:
        return A[0][0]

    if n == 2:
        diag_p = []

        for i in range(n):
            diag_p.append(A[i][i])

        diag_s = []

        for i in range(n):
            for j in range(n):
                if i + j == n - 1:
                    diag_s.append(A[i][j])

        result_p = 1
        for x in diag_p:
            result_p *= x

        result_s = 1
        for x in diag_s:
            result_s *= x

        return result_p - result_s

    t_s, _ = upper_t(A)

    r = 1
    for i in range(n):
        r = r * t_s[i][i]

    return r


def inversed(A):
    return normalize(A)[1]


def mul(A, B):
    rows_a = len(A)
    cols_a = len(A[0])

    rows_b = len(B)
    cols_b = len(B[0])

    # A(m x n) * B(n x p)
    if cols_a != rows_b:
        raise Exception('Não consigo multiplicar.')

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for m in range(rows_a):
        for p in range(cols_b):
            for n in range(cols_a):
                result[m][p] += A[m][n] * B[n][p]

    return result


def identity(A):
    n = len(A)

    r = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                r[i][j] = 1

    return r


def avg(A):
    return sum(A) / len(A)


# Função de correlação da regressão linear
def corr(X: list[int], y: list[float]) -> float:
    avg_x = avg(X)
    avg_y = avg(y)

    de: float = 0.0
    dv1: float = 0.0
    dv2: float = 0.0

    for x, y_ in zip(X, y):
        de += (x - avg_x) * (y_ - avg_y)
        dv1 += (x - avg_x)**2
        dv2 += (y_ - avg_y)**2

    return de / math.sqrt(dv1 * dv2)


# Função de regressão linear, retorna os betas + ŷ
def lin_reg(x_: float, X: list[int], y: list[float]):
    avg_x = avg(X)
    avg_y = avg(y)

    b1: float = 0.0

    de: float = 0.0
    dv: float = 0.0

    for x, y_ in zip(X, y):
        de += (x - avg_x) * (y_ - avg_y)
        dv += (x - avg_x)**2

    b1 = de / dv

    b0 = avg_y - (b1 * avg_x)

    y_pred = b0 + (b1 * x_)

    return b0, b1, y_pred


# Função de regressão linear múltipla, retorna os betas + ŷ
def lin_reg_mul(X, y):
    def B(X, y):
        x_T = transpose(X)
        c = mul(x_T, X)
        c_inversa = inversed(c)
        r = mul(c_inversa, x_T)
        return mul(r, y)

    betas = B(X, y)

    return betas, predict(X, betas)


# Função de previsão, retorna os valores previstos para os dados de entrada X e os betas calculados.
def predict(X, betas):
    return mul(X, betas)


# Função de erro quadrático, retorna o SSE, que é a soma dos quadrados dos erros,
# ou seja, a soma das diferenças entre os valores reais e os valores previstos ao quadrado.

# É utilizado posteriormente pelo polyfit para calcular o MSE, que é o erro médio quadrático.
def sse(x, y, coefs):
    s = 0

    for xi, yi in zip(x, y):
        y_pred = 0

        for exp, b in enumerate(coefs):
            y_pred += b * (xi ** exp)

        s += (yi - y_pred) ** 2

    return s


# Função de erro médio quadrático, retorna o MSE, que é o erro médio quadrático,
# ou seja, a média das diferenças entre os valores reais e os valores previstos ao quadrado.
def mse(x, y):
    n = len(x)
    b0, b1, _ = lin_reg(x[0], x, y)
    return sse(x, y, [b0, b1]) / n


# Função de resolução de sistemas lineares, retorna a solução do sistema Ax = b.
def solve(A, b):
    n = len(A)

    M = [
        A[i][:] + [b[i]]
        for i in range(n)
    ]

    for col in range(n):
        pivot = col

        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[pivot][col]):
                pivot = row

        M[col], M[pivot] = M[pivot], M[col]

        divisor = M[col][col]

        for j in range(col, n + 1):
            M[col][j] /= divisor

        for row in range(n):
            if row == col:
                continue

            factor = M[row][col]

            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    return [M[i][-1] for i in range(n)]


# Função de ajuste polinomial, retorna os coeficientes do polinômio que melhor se ajusta aos dados.
# É equivalente ao numpy.polyfit, mas implementado manualmente para fins de aprendizado.
def polyfit(x, y, deg):
    X = []

    for xi in x:
        row = []

        for power in range(deg + 1):
            row.append(xi ** power)

        X.append(row)

    XT = transpose(X)

    XTX = mul(XT, X)

    Y = [[yi] for yi in y]
    XTY = mul(XT, Y)

    rhs = [row[0] for row in XTY]

    coefs = solve(XTX, rhs)

    return coefs


# Demonstração da 1ª parte do trabalho, que consiste em regressão linear simples.
def demo_linear_regression():
    import matplotlib.pyplot as plt
    import pandas as pd


    # Pega os dados do arquivo CSV e retorna um par de cor (laranja, vermelho, etc.) e pontos (X, y)
    def get_datasets():
        R = pd.read_csv("trabalho_1_parte_1.csv")

        return [
            {
                "color": "red",
                "points": (R["x1"].tolist(), R["y1"].tolist()),
            },
            {
                "color": "blue",
                "points": (R["x2"].tolist(), R["y2"].tolist()),
            },
            {
                "color": "green",
                "points": (R["x3"].tolist(), R["y3"].tolist()),
            },
            {
                "color": "orange",
                "points": (R["x4"].tolist(), R["y4"].tolist()),
            }
        ]


    # Mostra um gráfico para o par X, Y e plota a linha de regressão linear.
    def show_chart(d):
        x = d['points'][0]
        y = d['points'][1]
        color = d['color']

        y_pred = []

        c = corr(x, y)

        for x_ in x:
            b0, b1, r = lin_reg(x_, x, y)
            y_pred.append(r)

            title = f"Correlação: {c:.4f}, y={b0:.4f}+{b1:.4f}*X"
            plt.title(title)

        plt.scatter(x, y, color=color, label='Dataset')

        plt.plot(x, y_pred)

        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()

    for dataset in get_datasets():
        show_chart(dataset)


# Demonstração da 2ª parte do trabalho, que consiste em regressão linear múltipla.
def demo_multiple_regression():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Pega os dados do arquivo CSV e retorna X e y.
    def get_dataset():
        R = pd.read_csv("trabalho_1_parte_2.csv", header=None) \
            .to_numpy() \
            .tolist()

        RESULTS_COL_IDX = 2
        X_1_COL_IDX = 0
        X_2_COL_IDX = 1

        X = []
        y = []

        for i in range(len(R)):
            X.append([])
            X[i].insert(0, 1)
            X[i].insert(1, R[i][X_1_COL_IDX])
            X[i].insert(2, R[i][X_2_COL_IDX])

            y.append([])
            y[i].append(R[i][RESULTS_COL_IDX])

        return X, y

    # Exibe a correlação e a regressão linear simples de uma variável em relação ao preço da casa.
    def show_simple_regression(
        values,
        prices,
        prediction_value,
        correlation_message,
        regression_message,
        x_label,
        chart_title,
    ):
        print(correlation_message)
        c = corr(values, prices)
        print(c)

        print(regression_message)
        b0, b1, _ = lin_reg(prediction_value, values, prices)
        print(f"b0: {b0}, b1: {b1}")

        plt.scatter(values, prices, label="y")
        plt.plot(
            values,
            [b0 + b1 * x for x in values],
            color='red',
            label="Linha de regressão",
        )

        plt.xlabel(x_label)
        plt.ylabel("Preço da casa (R$)")
        plt.title(f"{chart_title}, correlação: {c:.2f}")
        plt.legend()
        plt.show()

    X, y = get_dataset()

    house_sizes = [x[1] for x in X]
    house_prices = [y[0] for y in y]
    house_bedrooms = [x[2] for x in X]

    # Análise individual: tamanho da casa x preço.
    show_simple_regression(
        house_sizes,
        house_prices,
        2000,
        "Correlação: Tamanho casa x Preço casa:",
        "Regressão linear: Tamanho casa x Preço casa:",
        "Tamanho da casa (sq ft)",
        "Tamanho da casa vs Preço da casa",
    )

    # Análise individual: quantidade de quartos x preço.
    show_simple_regression(
        house_bedrooms,
        house_prices,
        3,
        "Correlação: Quantidade quartos x Preço casa:",
        "Regressão linear: Quantidade quartos x Preço casa:",
        "Quantidade de quartos",
        "Quantidade de quartos vs Preço da casa",
    )

    # Regressão linear múltipla e visualização do plano de regressão.

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

    betas, _ = lin_reg_mul(X, y)

    grid_X = [
        [1, size, bedrooms]
        for size, bedrooms in zip(size_grid.ravel(), bedroom_grid.ravel())
    ]

    price_grid = predict(grid_X, betas)

    price_grid = np.array([
        value[0] for value in price_grid
    ]).reshape(size_grid.shape)

    ax.plot_surface(
        size_grid,
        bedroom_grid,
        price_grid,
        alpha=0.6,
    )  # Plano de regressão

    ax.set_xlabel("Tamanho da casa")
    ax.set_ylabel("Quantidade de quartos")
    ax.set_zlabel("Preço da casa (R$)")

    ax.set_title(
        "Tamanho da casa e quantidade de quartos vs. preço da casa"
    )

    plt.show()

    # Previsões usando os coeficientes calculados manualmente.
    def calc(size, bedrooms):
        return betas[0][0] + betas[1][0] * size + betas[2][0] * bedrooms

    print(f"Preço esperado para uma casa de 1650 e 3 quartos: R${calc(1650, 3):.2f}")
    print(f"Preço esperado para uma casa de 1650 e 2 quartos: R${calc(1650, 2):.2f}")
    print(f"Preço esperado para uma casa de 1650 e 4 quartos: R${calc(1650, 4):.2f}")

    # Comparação com a implementação do scikit-learn.
    from sklearn.linear_model import LinearRegression

    lib_model = LinearRegression()
    lib_model.fit(X, y)

    sklearn_predict = lib_model.predict([[1, 1650, 3]])[0][0]

    print(f"Preço esperado para uma casa de 1650 e 3 quartos (sklearn): R${sklearn_predict:.2f}")


# Demonstração da 3ª parte do trabalho, que consiste em regressão polinomial.
def demo_polynomial_regression():
    import matplotlib.pyplot as plt
    import pandas as pd


    # Define o "blueprint" dos gráficos a serem plotados, que consiste em uma lista de tuplas (grau do polinômio, cor do gráfico)
    PLOTS_BLUEPRINT = [(1, 'red'), (2, 'green'), (3, 'black'), (8, 'yellow')]

    # Pega os dados do arquivo CSV e retorna X e y
    def get_datasets():
        R = pd.read_csv("trabalho_1_parte_3.csv", header=None)

        X = R[0].tolist()
        y = R[1].tolist()

        return X, y

    # Dividir aleatoriamente sendo 10% dos dados para teste e 90% para treino
    def spread(x, y, test_p=0.1, seed=42):
        import random

        n = len(x)

        idxs = list(range(n))

        random.seed(seed)
        random.shuffle(idxs)

        n_test = int(n * test_p)

        if n_test == 0:
            n_test = 1

        idxs_test = idxs[:n_test]
        idxs_train = idxs[n_test:]

        x_train = []
        y_train = []

        for i in idxs_train:
            x_train.append(x[i])
            y_train.append(y[i])

        x_test = []
        y_test = []

        for i in idxs_test:
            x_test.append(x[i])
            y_test.append(y[i])

        return x_train, y_train, x_test, y_test

    def show_chart(x_train, y_train, x_test, y_test):
        from sklearn.metrics import r2_score

        plt.scatter(x_train, y_train, color='blue', label='Treino')
        plt.scatter(x_test, y_test, color='#FF1493', marker='x', linewidths=2, label='Teste')

        results = []

        for deg, color in PLOTS_BLUEPRINT:
            coefs = polyfit(x_train, y_train, deg)

            sorted_x_train = sorted(x_train)

            y_curva = []
            for x_ in sorted_x_train:
                y_ = 0

                for power, beta in enumerate(coefs):
                    y_ += beta * (x_ ** power)

                y_curva.append(y_)

            plt.plot(sorted_x_train, y_curva, color=color)

            eqm_treino = sse(x_train, y_train, coefs) / len(x_train)
            eqm_teste = sse(x_test, y_test, coefs) / len(x_test)

            results.append((deg, eqm_teste)) 

            y_previsto_treino = []
            for x_ in x_train:
                y_ = 0
                for power, beta in enumerate(coefs):
                    y_ += beta * (x_ ** power)
                y_previsto_treino.append(y_)

            y_previsto_teste = []
            for x_ in x_test:
                y_ = 0
                for power, beta in enumerate(coefs):
                    y_ += beta * (x_ ** power)
                y_previsto_teste.append(y_)
            r2_treino = r2_score(y_train, y_previsto_treino)
            r2_teste = r2_score(y_test, y_previsto_teste)

            print(
                f"Treino-Teste Grau {deg} / "
                f"EQM p treino {eqm_treino:.4f} / EQM p teste {eqm_teste:.4f} / "
                f"R2 treino {r2_treino:.4f} / R2 teste {r2_teste:.4f}"
            )
        melhor_grau, menor_eqm_teste = min(results, key=lambda r: r[1])

        print(f"k)Modelo mais preciso nos dados de teste é o de grau {melhor_grau}")

        plt.title("Regressão polinomial só c dados de treino")
        plt.legend()

        plt.show()

    X, y = get_datasets()

    x_train, y_train, x_test, y_test = spread(X, y)

    show_chart(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    # Aqui, você pode escolher qual demonstração deseja executar.
    # Descomente a linha correspondente à demonstração desejada.

    # demo_linear_regression()
    # demo_multiple_regression()
    demo_polynomial_regression()
