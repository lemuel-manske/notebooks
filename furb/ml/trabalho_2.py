import statistics
import math

import scipy.io as scipy


DATASET_1 = scipy.loadmat('trabalho_2_parte_1_1.mat')
DATASET_2 = scipy.loadmat('trabalho_2_parte_1_2.mat')
DATASET_3 = scipy.loadmat('trabalho_2_parte_1_3.mat')
DATASET_4 = scipy.loadmat('trabalho_2_parte_1_4.mat')


type Vector = list[int]

def d(x: Vector, y: Vector) -> float:
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


def knn(
    train: list[list[int]],  # 100 x 4
    labels: list[str],  # 100 x 1
    test: list[list[int]],  # 100 x 4
    k: int,
) -> list[str]:
    pred = []
    for t in test:
        dists = []
        for ti in range(len(train)):
            dists.append((ti, d(train[ti], t)))
        dists.sort(key=lambda d_: d_[1])
        nn = dists[:k]
        pred.append(labels[nn[0][0]])  # Política de decisão: sempre retornar o primeiro
    return max(set(pred), key=pred.count)


def demo_knn(dataset, k):
    res = knn(
        dataset['grupoTrain'],
        dataset['trainRots'],
        dataset['grupoTest'],
        k,
    )

    print(
        sum(res == DATASET_1['testRots']) / len(DATASET_1['testRots'])
    )


if __name__ == "__main__":
    demo_knn(DATASET_1, k=1)
    demo_knn(DATASET_1, k=10)
