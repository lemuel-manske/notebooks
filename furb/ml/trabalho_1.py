import math
import copy


def transpose(A) :
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


# Substituição regressiva, que é utilizada para resolver sistemas lineares.
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


# Função de resolução de sistemas lineares, retorna a solução do sistema Ax = b.
# Usamos para resolver a regressão polinomial.
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


# Função de regressão linear, retorna os betas + y previsto
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


# Função de previsão, retorna os valores previstos para os dados de entrada X e os betas calculados.
def predict(X, betas):
    return mul(X, betas)


# Função de regressão linear múltipla, retorna os betas + y previsto
def lin_reg_mul(X, y):
    def B(X, y):
        x_T = transpose(X)
        c = mul(x_T, X)
        c_inversa = inversed(c)
        r = mul(c_inversa, x_T)
        return mul(r, y)

    betas = B(X, y)

    return betas, predict(X, betas)


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


# Demonstração da 1a parte do trabalho, que consiste em regressão linear simples.
def demo_linear_regression():
    import pandas as pd


    # Pega os dados do arquivo CSV e retorna um par de cor (laranja, vermelho, etc.) e pontos (X, y)
    def get_datasets():
        R = pd.read_csv("trabalho_1_parte_1.csv")

        return [
            {
                "title": "Dataset 1",
                "color": "red",
                "points": (R["x1"].tolist(), R["y1"].tolist()),
            },
            {
                "title": "Dataset 2",
                "color": "blue",
                "points": (R["x2"].tolist(), R["y2"].tolist()),
            },
            {
                "title": "Dataset 3",
                "color": "green",
                "points": (R["x3"].tolist(), R["y3"].tolist()),
            },
            {
                "title": "Dataset 4",
                "color": "orange",
                "points": (R["x4"].tolist(), R["y4"].tolist()),
            }
        ]


    # Mostra um gráfico para o par X, Y e plota a linha de regressão linear para um dado dataset.
    def show_chart(d):
        import matplotlib.pyplot as plt

        x = d['points'][0]
        y = d['points'][1]
        color = d['color']

        y_pred = []

        c = corr(x, y) # correlação

        for x_ in x:
            b0, b1, r = lin_reg(x_, x, y) # regressão linear
            y_pred.append(r)

            title = f"Correlação: {c:.4f}, y={b0:.4f}+{b1:.4f}*X"
            plt.title(title)

        plt.scatter(x, y, color=color, label=d['title'])

        plt.plot(x, y_pred)

        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()

    for dataset in get_datasets():
        show_chart(dataset)

    print()
    print(
        "3) Qual dos datasets não é apropriado para regressão linear? Justifique sua resposta."
    )

    print(
        "O dataset 2 (de cor azul) não é apropriado para regressão linear. ",
        "Seria um caso de regressão polinomial, pois a relação entre X e Y não é linear. ",
        "Nesse caso, estamos usando um chinelo para matar um dinossauro.",
    )

    print()
    print(
        "4) Ao analisar o gráfico de dispersão e o resultado da regressão linear para o dataset 4, observa-se um problema. O que deveria ser feito antes de ajustar o modelo de regressão? Justifique sua resposta."
    )

    print(
        "O dataset 4 possui um outlier, que é um ponto de dados que se distancia significativamente dos demais. ",
        "Antes de ajustar o modelo de regressão, seria importante identificar e tratar esse outlier (removendo ou analisando melhor). ",
        "Para demonstrar, removemos o outlier e refizemos a regressão linear, obtendo uma linha de regressão mais adequada aos dados restantes.",
    )

    dataset_4 = get_datasets()[3]

    outlier_idx = 2
    dataset_4_wo_outlier = {
        "title": "Dataset 4 (sem outlier)",
        "color": "orange",
        "points": (
            dataset_4['points'][0][:outlier_idx] + dataset_4['points'][0][outlier_idx + 1:],
            dataset_4['points'][1][:outlier_idx] + dataset_4['points'][1][outlier_idx + 1:],
        )
    }

    show_chart(dataset_4_wo_outlier)


# Demonstração da 2a parte do trabalho, que consiste em regressão linear múltipla.
def demo_multiple_regression():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Pega os dados do arquivo CSV e retorna X e y.
    def get_dataset():
        R = pd.read_csv("trabalho_1_parte_2.csv", header=None)

        print()
        print("b) Utilize o comando python .describe() para fazer uma primeira análise estatística da sua base de dados. Qual a média de preço das casas? Quanto custa a menor casa? Quantos quartos tem a casa mais cara?")

        print()
        print(R.describe())

        most_cheap = R[2].min()
        most_expensive_bedrooms = R[1].max()

        print()
        print(f"A menor casa custa {most_cheap:.2f}, e a casa mais cara tem {most_expensive_bedrooms:.0f} quartos.")

        R = R.to_numpy() \
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
        print()
        print(correlation_message)

        c = corr(values, prices)
        print(c)

        print()
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
        plt.ylabel("Preço da casa")

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
        "d) Tamanho da casa vs Preço da casa",
    )

    # Análise individual: quantidade de quartos x preço.
    show_simple_regression(
        house_bedrooms,
        house_prices,
        3,
        "Correlação: Quantidade quartos x Preço casa:",
        "Regressão linear: Quantidade quartos x Preço casa:",
        "Quantidade de quartos",
        "d) Quantidade de quartos vs Preço da casa",
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

    # Plano de regressão
    ax.plot_surface(
        size_grid,
        bedroom_grid,
        price_grid,
        alpha=0.6,
    )

    ax.set_xlabel("Tamanho da casa")
    ax.set_ylabel("Quantidade de quartos")
    ax.set_zlabel("Preço da casa")

    corr_house_sizes = corr(house_sizes.tolist(), house_prices.tolist())
    corr_house_bedrooms = corr(house_bedrooms.tolist(), house_prices.tolist())

    ax.set_title(
        f"e) Tamanho da casa e quantidade de quartos vs. preço da casa\n"
        f"Correlação tamanho casa x preço casa: {corr_house_sizes:.2f}\n"
        f"Correlação quantidade quartos x preço casa: {corr_house_bedrooms:.2f}"
    )

    plt.show()

    # Previsões usando os coeficientes calculados manualmente.
    def calc(size, bedrooms):
        return betas[0][0] + betas[1][0] * size + betas[2][0] * bedrooms

    print()
    print("h) Calcule o preço de uma casa que tem tamanho de 1650 e 3 quartos. O resultado deve ser igual a 293081. Aumente e diminua a quantidade de número de quartos. O que acontece? Por qual motivo?")

    print(f"Preço esperado para uma casa de 1650 e 3 quartos: {calc(1650, 3):.2f}")
    print(f"Preço esperado para uma casa de 1650 e 2 quartos: {calc(1650, 2):.2f}")
    print(f"Preço esperado para uma casa de 1650 e 4 quartos: {calc(1650, 4):.2f}")

    print()
    print(
        "Isso acontece porque o coeficiente associado ao número de quartos na regressão múltipla é negativo. ",
        "Mesmo que mais quartos geralmente aumentem o preço, quando o tamanho da casa é mantido igual, o modelo indica que mais quartos podem diminuir o valor estimado devido à relação entre tamanho e quantidade de quartos nos dados."
    )

    # Comparação com a implementação do scikit-learn.
    from sklearn.linear_model import LinearRegression

    lib_model = LinearRegression()
    lib_model.fit(X, y)

    sklearn_predict = lib_model.predict([[1, 1650, 3]])[0][0]

    print()
    print("i) Compare seus resultados com a função de regressão linear múltipla do python. Para isso você irá precisar das bibliotecas numpy e scikit-learn.")
    print(f"Preço esperado para uma casa de 1650 e 3 quartos (sklearn): {sklearn_predict:.2f}")


# Demonstração da 3a parte do trabalho, que consiste em regressão polinomial.
def demo_polynomial_regression():
    import random

    import matplotlib.pyplot as plt
    import pandas as pd

    from sklearn.metrics import r2_score

    PLOTS_BLUEPRINT = [
        (1, "red"),
        (2, "green"),
        (3, "black"),
        (8, "yellow"),
    ]

    # Pega os dados do arquivo CSV e retorna X e y.
    def get_dataset():
        R = pd.read_csv("trabalho_1_parte_3.csv", header=None)

        X = R[0].tolist()
        y = R[1].tolist()

        return X, y

    # Calcula o valor de y previsto para um valor de x e os coeficientes do polinômio.
    def calc(x, coefs):
        y_pred = 0

        for power, beta in enumerate(coefs):
            y_pred += beta * (x ** power)

        return y_pred

    # Divide aleatoriamente os dados, sendo 10% para teste e 90% para treino.
    def split_dataset(X, y, test_percentage=0.1, seed=47):
        indexes = list(range(len(X)))

        random.seed(seed)
        random.shuffle(indexes)

        test_size = max(1, int(len(X) * test_percentage))

        test_indexes = indexes[:test_size]
        train_indexes = indexes[test_size:]

        x_train = [X[i] for i in train_indexes]
        y_train = [y[i] for i in train_indexes]

        x_test = [X[i] for i in test_indexes]
        y_test = [y[i] for i in test_indexes]

        return x_train, y_train, x_test, y_test

    X, y = get_dataset()

    # b) Gráfico de dispersão dos dados.
    plt.scatter(
        X,
        y,
        label="Dados observados",
    )

    sorted_X = sorted(X)

    errors = []

    print()
    print(
        "g) Calcule o Erro Quadrático Médio (EQM) para cada linha de regressão. "
        "Qual é o mais preciso?"
    )

    # c - f) Calcula e plota as regressões de graus 1, 2, 3 e 8.
    for degree, color in PLOTS_BLUEPRINT:
        coefs = polyfit(
            X,
            y,
            degree,
        )

        y_curve = [
            calc(x, coefs)
            for x in sorted_X
        ]

        plt.plot(
            sorted_X,
            y_curve,
            color=color,
            label=f"Grau {degree}",
        )

        eqm = sse(
            X,
            y,
            coefs,
        ) / len(X)

        errors.append(
            (degree, eqm)
        )

        print(
            f"Grau {degree}: EQM = {eqm:.4f}"
        )

    best_degree, best_eqm = min(
        errors,
        key=lambda item: item[1],
    )

    print()
    print(
        f"Considerando somente o EQM calculado sobre todos os dados, "
        f"o modelo mais preciso é o de grau {best_degree}, "
        f"com EQM = {best_eqm:.4f}."
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        "b - g) Regressão polinomial"
    )

    plt.legend()
    plt.show()

    # h) Divide aleatoriamente os dados em treino e teste.
    x_train, y_train, x_test, y_test = split_dataset(
        X,
        y,
    )

    print()
    print(
        "h) Os dados foram divididos aleatoriamente em 90% para treinamento "
        "e 10% para teste."
    )

    print(
        f"Quantidade de dados de treino: {len(x_train)}"
    )

    print(
        f"Quantidade de dados de teste: {len(x_test)}"
    )

    # i) Ajusta novamente as regressões usando somente os dados de treinamento.
    plt.scatter(
        x_train,
        y_train,
        label="Treino",
    )

    plt.scatter(
        x_test,
        y_test,
        color="magenta",
        marker="x",
        linewidths=2,
        label="Teste",
    )

    results = []

    print()
    print(
        "j) Calcule o EQM utilizando somente os dados de teste."
    )

    for degree, color in PLOTS_BLUEPRINT:
        coefs = polyfit(
            x_train,
            y_train,
            degree,
        )

        y_curve = [
            calc(x, coefs)
            for x in sorted_X
        ]

        plt.plot(
            sorted_X,
            y_curve,
            color=color,
            label=f"Grau {degree}",
        )

        y_pred_train = [
            calc(x, coefs)
            for x in x_train
        ]

        y_pred_test = [
            calc(x, coefs)
            for x in x_test
        ]

        eqm_train = sse(
            x_train,
            y_train,
            coefs,
        ) / len(x_train)

        eqm_test = sse(
            x_test,
            y_test,
            coefs,
        ) / len(x_test)

        r2_train = r2_score(
            y_train,
            y_pred_train,
        )

        r2_test = r2_score(
            y_test,
            y_pred_test,
        )

        results.append(
            (
                degree,
                eqm_train,
                eqm_test,
                r2_train,
                r2_test,
            )
        )

        print()
        print(
            f"Grau {degree}:"
        )

        print(
            f"EQM treino: {eqm_train:.4f}"
        )

        print(
            f"EQM teste: {eqm_test:.4f}"
        )

        print(
            f"R2 treino: {r2_train:.4f}"
        )

        print(
            f"R2 teste: {r2_test:.4f}"
        )

    print()
    print(
        "k) Calcule o R2 para os dados de treino e teste. "
        "O que se pode concluir com os resultados?"
    )

    print(
        "Quanto mais próximo de 1 for o R2, melhor o modelo consegue explicar "
        "a variação dos dados. Porém, um R2 muito alto nos dados de treino e "
        "consideravelmente menor nos dados de teste pode indicar overfitting."
    )

    best_model = min(
        results,
        key=lambda item: item[2],
    )

    print()
    print(
        "l) Visto o cálculo do erro e do coeficiente de determinação, "
        "qual o modelo mais preciso neste caso?"
    )

    print(
        f"Considerando principalmente o desempenho nos dados de teste, "
        f"o modelo mais preciso é o de grau {best_model[0]}, "
        f"com EQM de teste = {best_model[2]:.4f} "
        f"e R2 de teste = {best_model[4]:.4f}."
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        "i - l) Regressão polinomial com dados de treino e teste"
    )

    plt.legend()
    plt.show()

    # m) Calcula a regressão polinomial de grau 20.
    degree = 20

    coefs = polyfit(
        x_train,
        y_train,
        degree,
    )

    y_pred_train = [
        calc(x, coefs)
        for x in x_train
    ]

    y_pred_test = [
        calc(x, coefs)
        for x in x_test
    ]

    eqm_train_20 = sse(
        x_train,
        y_train,
        coefs,
    ) / len(x_train)

    eqm_test_20 = sse(
        x_test,
        y_test,
        coefs,
    ) / len(x_test)

    r2_train_20 = r2_score(
        y_train,
        y_pred_train,
    )

    r2_test_20 = r2_score(
        y_test,
        y_pred_test,
    )

    print()
    print(
        "m) Trace a curva de regressão polinomial de grau 20 e compare seu "
        "comportamento com os modelos de grau 1, 2, 3 e 8."
    )

    print()
    print(
        "Grau 20:"
    )

    print(
        f"EQM treino: {eqm_train_20:.4f}"
    )

    print(
        f"EQM teste: {eqm_test_20:.4f}"
    )

    print(
        f"R2 treino: {r2_train_20:.4f}"
    )

    print(
        f"R2 teste: {r2_test_20:.4f}"
    )

    print()
    print(
        "O modelo de grau 20 tende a se ajustar muito bem aos dados de treino, "
        "mas sua curva apresenta oscilações maiores e pode ter desempenho pior "
        "nos dados de teste. Isso ocorre por causa do overfitting: o modelo "
        "fica complexo demais e passa a representar particularidades dos dados "
        "de treinamento em vez de representar apenas a tendência geral (generalização)."
    )

    plt.scatter(
        x_train,
        y_train,
        label="Treino",
    )

    plt.scatter(
        x_test,
        y_test,
        color="magenta",
        marker="x",
        linewidths=2,
        label="Teste",
    )

    # Plota novamente os graus 1, 2, 3 e 8 para facilitar a comparação.
    for degree, color in PLOTS_BLUEPRINT:
        coefs = polyfit(
            x_train,
            y_train,
            degree,
        )

        y_curve = [
            calc(x, coefs)
            for x in sorted_X
        ]

        plt.plot(
            sorted_X,
            y_curve,
            color=color,
            label=f"Grau {degree}",
        )

    coefs_20 = polyfit(
        x_train,
        y_train,
        20,
    )

    y_curve_20 = [
        calc(x, coefs_20)
        for x in sorted_X
    ]

    plt.plot(
        sorted_X,
        y_curve_20,
        color="purple",
        label="Grau 20",
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(
        "m) Comparação com regressão polinomial de grau 20"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Aqui, você pode escolher qual demonstração deseja executar.
    # Descomente a linha correspondente à demonstração desejada.

    # demo_linear_regression()
    # demo_multiple_regression()
    demo_polynomial_regression()
