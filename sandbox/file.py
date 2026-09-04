import copy
import pandas as pd


R = pd.read_csv("data.csv", header=None)
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


def run_demo():
    X = get_data()
    y = get_expected_results()

    print("X:")
    print(X)
    print("y:")
    print(y)

    b = B(X, y)
    print("B:")
    print(b)

    r = run_multiple_regression(X, y)
    print("R:")
    print(r)


if __name__ == "__main__":
    run_demo()
