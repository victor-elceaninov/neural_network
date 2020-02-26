import numpy


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + numpy.exp(-x))


training_inputs = numpy.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

training_outputs = numpy.array([[0, 1, 1, 0]]).T

# сделаем случайные числа более определёнными
numpy.random.seed(1)

# инициализируем веса случайным образом со средним 0
synaptic_weights = 2 * numpy.random.random((3, 1)) - 1
print(f'Random starting synaptic weights: \n {synaptic_weights} \n')

outputs = None

for iteration in range(60000):
    # прямое распространение
    input_layer = training_inputs
    outputs = sigmoid(numpy.dot(input_layer, synaptic_weights))

    # насколько мы ошиблись?
    error = training_outputs - outputs

    # перемножим это с наклоном сигмоиды
    # на основе значений в second_neural_layout
    adjustments = error * sigmoid(outputs, derivative=True)  # !!!

    # обновим веса
    synaptic_weights += numpy.dot(input_layer.T, adjustments)  # !!!

print(f'Synoptic weight after training: \n {synaptic_weights} \n')
print(f'Outputs after training: \n{outputs} \n')
