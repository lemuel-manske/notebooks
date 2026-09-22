import math

import scipy.io as scipy


DATASET_1 = scipy.loadmat('trabalho_2_parte_1_1.mat')
DATASET_2 = scipy.loadmat('trabalho_2_parte_1_2.mat')
DATASET_3 = scipy.loadmat('trabalho_2_parte_1_3.mat')
DATASET_4 = scipy.loadmat('trabalho_2_parte_1_4.mat')


def d(x, y) -> float:
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


def mode(lst):
    return max(lst, key=lambda x: (lst.count(x), -lst.index(x)))


def kNN(
    train,
    labels,
    test,
    k: int,
) -> list:
    pred = []
    for t in test:
        dists = []
        for ti in range(len(train)):
            dists.append((ti, d(train[ti], t)))
        dists.sort(key=lambda d_: d_[1])
        nn = dists[:k]
        nn_labels = [labels[i][0] for i, _ in nn]  # política de decisão: sempre retornar o primeiro
        pred.append(mode(nn_labels))
    return pred


def acc(nn, labels):
    labels = [
        label[0]
        for label in labels
    ]

    return sum(r == e for r, e in zip(nn, labels)) / len(labels)


def find_best_k(
    train,
    train_labels,
    test,
    test_labels,
) -> int:
    best_acc = 0.0
    best_k = 0
    for i in range(1, len(train)):
        curr_acc = acc(kNN(train, train_labels, test, i), test_labels)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_k = i
    return best_k


def min_max_normalizer(train_data, test_data):
    # Normaliza cada característica para o intervalo [0, 1] usando min/max do treino

    _len = train_data.shape[1]

    train = train_data.copy().astype(float)
    test = test_data.copy().astype(float)

    for col in range(_len):
        _min = train_data[:, col].min()
        _max = train_data[:, col].max()

        if _max - _min == 0:
            train[:, col] = 0
            test[:, col] = 0
            continue

        train[:, col] = (train_data[:, col] - _min) / (_max - _min)
        test[:, col] = (test_data[:, col] - _min) / (_max - _min)

    return train, test


def avg(data, idx):
    return sum(data[:, idx]) / len(data)


def std(data, idx):
    mean = avg(data, idx)
    return math.sqrt(sum((x - mean) ** 2 for x in data[:, idx]) / len(data))


def padronize(train_data, test_data):
    _len = train_data.shape[1]

    train = train_data.copy().astype(float)
    test = test_data.copy().astype(float)

    for col in range(_len):
        mean = avg(train_data, col)
        std_dev = std(train_data, col)

        if std_dev == 0:
            train[:, col] = 0
            test[:, col] = 0
            continue

        train[:, col] = (train_data[:, col] - mean) / std_dev
        test[:, col] = (test_data[:, col] - mean) / std_dev

    return train, test


def get_label(data, labels, label, idx):
    ret = []
    for i in range(len(data)):
        if(labels[i] == label):
            ret.append(data[i][idx])
    return ret


def show_chart(data, labels, d1, d2):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(get_label(data, labels, 1, d1), get_label(data, labels, 1, d2), c='red' , marker='^')
    ax.scatter(get_label(data, labels, 2, d1), get_label(data, labels, 2, d2), c='blue' , marker='+')
    ax.scatter(get_label(data, labels, 3, d1), get_label(data, labels, 3, d2), c='green', marker='.')
    plt.show()


def demo_1():
    grupoTrain = DATASET_1['grupoTrain']
    trainRots = DATASET_1['trainRots']
    grupoTest = DATASET_1['grupoTest']
    testRots = DATASET_1['testRots']

    show_chart(grupoTrain, trainRots, 1, 2)

    best_k = find_best_k(grupoTrain, trainRots, grupoTest, testRots)

    res = kNN(grupoTrain, trainRots, grupoTest, best_k)

    print()
    print(f"Q1.1 = melhor acurácia foi de: {acc(res, testRots)} com k = {best_k}")

    print()
    print("Q1.2")
    for removed_col in range(4):
        rest = [c for c in range(4) if c != removed_col]
        tr_sub = grupoTrain[:, rest]
        te_sub = grupoTest[:, rest]

        k_combo = find_best_k(tr_sub, trainRots, te_sub, testRots)

        k_res = kNN(tr_sub, trainRots, te_sub, k_combo)

        print(f"Sem a coluna {removed_col}: / melhor k = {k_combo} / acurácia = {acc(k_res, testRots):.2f}")

    print()
    print(
        "Testando o dataset sem cada característica, uma de cada vez,",
        "é possível remover a coluna 1 e a coluna 3 (comprimento e largura da sépala, respectivamente)",
        "sem perder acurácia, mantendo os mesmos 98% obtidos com todas as 4 características, então",
        "não é necessário ter todas as características pra obter a acurácia máxima.",
    )


def demo_2():
    grupoTrain = DATASET_2['grupoTrain']
    trainRots = DATASET_2['trainRots']
    grupoTest = DATASET_2['grupoTest']
    testRots = DATASET_2['testRots']

    show_chart(grupoTrain, trainRots, 0, 1)

    res = kNN(grupoTrain, trainRots, grupoTest, k=3)  # assumindo k=3 (melhor k, supostamente)

    print()
    print(f"Q2.1 = acurácia com k = 3: {acc(res, testRots):.5f}")

    train_normalized, test_normalized = min_max_normalizer(grupoTrain, grupoTest)

    print()
    print("Q2.2")
    for k in range(1, 15):
        res_k = kNN(train_normalized, trainRots, test_normalized, k)

        print(f"k = {k}: / acurácia normalizada = {acc(res_k, testRots):.2f}")

    print()
    print(
        "Foi usada a normalização devido as características do dataset estarem em escalas diferentes,",
        "visto que muitas características tem amplitude entre 3 e 5 no valor, e a prolina tem uma diferença de mais de 1000 unidades.",
    )


def demo_3():
    grupoTrain = DATASET_3['grupoTrain']
    trainRots = DATASET_3['trainRots']
    grupoTest = DATASET_3['grupoTest']
    testRots = DATASET_3['testRots']

    show_chart(grupoTrain, trainRots, 0, 1)

    res = kNN(grupoTrain, trainRots, grupoTest, k=1)

    print()
    print(f"Q3.1 = acurácia com k = 1: {acc(res, testRots)}")

    print()
    print("Q3.2 - Sem padronização:")
    for k in range(1, 15):
        res_k = kNN(grupoTrain, trainRots, grupoTest, k)

        print(f"k = {k}: / acurácia = {acc(res_k, testRots):.2f}")

    print()
    print("Q3.2 - Com padronização:")
    train_normalized, test_normalized = padronize(grupoTrain, grupoTest)
    for k in range(1, 15):
        kNN_res = kNN(train_normalized, trainRots, test_normalized, k)

        print(f"k = {k} / acurácia = {acc(kNN_res, testRots):.2f}")

    print()
    print(
        "Acurácia final com k = 10: 92%, a diferença de acurácia se deve ao fato da alteração do k,",
        "visto que com k = 1, depende apenas de 1 vizinho próximo, então um ponto na fronteira entre classes pode gerar",
        "uma classificação errada facilmente. Podemos observar isso no gráfico, com sinais como pontos de classes diferentes",
        "sobrepostos, e foi feito o ajuste para k = 10, após rodar um loop e identificar que era o menor valor que já retornava acurácia de 92%.",
    )


def demo_4():
    grupoTrain = DATASET_4['trainSet']
    trainRots = DATASET_4['trainLabs']
    grupoTest = DATASET_4['testSet']
    testRots = DATASET_4['testLabs']

    show_chart(grupoTrain, trainRots, 0, 1)

    res = kNN(grupoTrain, trainRots, grupoTest, 1)

    print()
    print(f"Q4.1 = acurácia com k = 1: {acc(res, testRots):.2f}")

    train_normalized, test_normalized = min_max_normalizer(grupoTrain, grupoTest)

    print()
    print("Q4.2 - Somente normalizando:")
    for k in range(1, 15):
        res_k = kNN(train_normalized, trainRots, test_normalized, k)

        print(f"k = {k}: / acurácia = {acc(res_k, testRots):.2f}")

    print()
    print("Q4.2 - Filtrando atributos de \"ruído\" e normalizando:")
    for k in range(1, 15):
        res_k = kNN(train_normalized[:, [0, 1]], trainRots, test_normalized[:, [0, 1]], k)

        print(f"k = {k}: / acurácia = {acc(res_k, testRots):.2f}")

    print()
    print(
        "A normalização min-max, isoladamente, não foi suficiente para atingir a acurácia desejada.",
        "Entretanto, após a remoção dos atributos com maior comportamento de",
        "ruído e utilizando apenas os dois primeiros atributos, o k=1 atingiu 93% de acurácia."
        "Isso mostra que a remoção de atributos irrelevantes ou ruidosos pode melhorar significativamente o desempenho do kNN.",
    )


if __name__ == "__main__":

    # demo_1()
    # demo_2()
    demo_3()
    # demo_4()
