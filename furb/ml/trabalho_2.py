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
    ]  # flatten

    return sum(r == e for r, e in zip(nn, labels)) / len(labels)


def find_best_k(
    train,
    train_labels,
    test,
    test_labels,
) -> int:
    best_acc = 0.0
    best_k = 0
    for i in range(1, len(train)):  # testar todos os `k` possíveis
        curr_acc = acc(kNN(train, train_labels, test, i), test_labels)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_k = i
    return best_k


def demo_knn(dataset):
    show_chart(dataset['grupoTrain'], dataset['trainRots'], 1, 2)

    print()

    best_k = find_best_k(
        dataset['grupoTrain'],
        dataset['trainRots'],
        dataset['grupoTest'],
        dataset['testRots'],
    )

    print(
        f"Best k: {best_k}"
    )

    res = kNN(
        dataset['grupoTrain'],
        dataset['trainRots'],
        dataset['grupoTest'],
        best_k,
    )

    print(
        f"Acc: {acc(res, dataset['testRots'])}"
    )


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


if __name__ == "__main__":
    demo_knn(DATASET_1)
