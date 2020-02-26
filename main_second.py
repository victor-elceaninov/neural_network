import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

training_outputs = np.array([[0, 1, 1, 0]]).T

# сделаем случайные числа более определёнными
np.random.seed(1)

# инициализируем веса случайным образом со средним 0
synaptic_weights_first = 2 * np.random.random((3, 4)) - 1
synaptic_weights_second = 2 * np.random.random((4, 1)) - 1

outputs_first = None
outputs_second = None

for iteration in range(60000):
    # прямое распространение
    input_layer = training_inputs
    outputs_first = sigmoid(np.dot(input_layer, synaptic_weights_first))
    outputs_second = sigmoid(np.dot(outputs_first, synaptic_weights_second))

    # как сильно мы ошиблись относительно нужной величины?
    error_second = training_outputs - outputs_second

    if (iteration % 10000) == 0:
        print(f'Error: {str(np.mean(np.abs(error_second)))}')

    # в какую сторону нужно двигаться?
    # если мы были уверены в предсказании, то сильно менять его не надо
    adjustments_second = error_second * sigmoid(outputs_second, derivative=True)

    # как сильно значения l1 влияют на ошибки в l2?
    error_first = adjustments_second.dot(synaptic_weights_second.T)

    # в каком направлении нужно двигаться, чтобы прийти к l1?
    # если мы были уверены в предсказании, то сильно менять его не надо
    adjustments_first = error_first * sigmoid(outputs_first, derivative=True)

    synaptic_weights_second += outputs_first.T.dot(adjustments_second)
    synaptic_weights_first += input_layer.T.dot(adjustments_first)

print(f'Outputs First after training: \n{outputs_first} \n')
print(f'Outputs Second after training: \n{outputs_second} \n')
