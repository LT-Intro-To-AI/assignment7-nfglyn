from neural import NeuralNet

print("\n\nTraining SQ\n\n")
sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
]
sqn = NeuralNet(1, 6, 1)
sqn.train(sq_training_data)

print()
print(sqn.test_with_expected(sq_training_data))
print(sqn.evaluate([0.66]))
print(sqn.evaluate([0.95]))

print("\n\nTraining XOR\n\n")
xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

xorn = NeuralNet(2, 1, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))


x_or_trainingdata = [
    ([0,0], [0]),
    ([1,0], [0]),
    ([0,1], [1]),
    ([1,1], [0])
]

xorn = NeuralNet(2, 20, 1)

xorn.train(x_or_trainingdata)

print()

print(xorn.test_with_expected(x_or_trainingdata))