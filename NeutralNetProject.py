from neural import NeuralNet

poker = NeuralNet(10, 6, 1)

poker_hard_training = [
    ([1, 10, 1, 11, 1, 13, 1, 12, 1, 1], [9]),
    ([2, 11, 2, 13, 2, 10, 2, 12, 2, 1], [9]),
    ([3, 12, 3, 11, 3, 13, 3, 10, 3, 1], [9]),
    ([4, 10, 4, 11, 4, 1, 4, 13, 4, 12], [9]),
    ([4, 1, 4, 13, 4, 12, 4, 11, 4, 10], [9]),
    ([1, 2, 1, 4, 1, 5, 1, 3, 1, 6], [8]),
    ([1, 9, 1, 12, 1, 10, 1, 11, 1, 13], [8]),
    ([2, 1, 2, 2, 2, 3, 2, 4, 2, 5], [8]),
    ([3, 5, 3, 6, 3, 9, 3, 7, 3, 8], [8]),
    ([4, 1, 4, 4, 4, 2, 4, 3, 4, 5], [8]),
    ([1, 1, 2, 1, 3, 9, 1, 5, 2, 3], [1]),
    ([2, 6, 2, 1, 4, 13, 2, 4, 4, 9], [0]),
    ([1, 10, 4, 6, 1, 2, 1, 1, 3, 8], [0]),
    ([2, 13, 2, 1, 4, 4, 1, 5, 2, 11], [0]),
    ([3, 8, 4, 12, 3, 9, 4, 2, 3, 2], [1])

]

test_data = [
[1, 1, 1, 13, 2, 4, 2, 3, 1, 12],
[3, 12, 3, 2, 3, 11, 4, 5, 2, 5],
[1, 9, 4, 6, 1, 4, 3, 2, 3, 9],
[1, 4, 3, 13, 2, 13, 2, 1, 3, 6],
[3, 10, 2, 7, 1, 2, 2, 11, 4, 9],
[1, 3, 4, 5, 3, 4, 1, 12, 4, 6],
[2, 6, 4, 11, 2, 3, 4, 9, 1, 7],
[3, 2, 4, 9, 3, 7, 4, 3, 4, 5],
[4, 4, 3, 13, 1, 8, 3, 9, 3, 10],
[1, 9, 3, 8, 4, 4, 1, 7, 3, 5],
]

answer_test = [
    [0],
    [1], 
    [1], 
    [1], 
    [0], 
    [0], 
    [0], 
    [0], 
    [0], 
    [0]
]
poker.train(poker_hard_training)

print(f"case 1: {test_data[0]} evaluates to {poker.evaluate(test_data[0])} actual result: {answer_test[0]}")
print(f"case 2: {test_data[1]} evaluates to {poker.evaluate(test_data[1])} actual result: {answer_test[1]}")
print(f"case 3: {test_data[2]} evaluates to {poker.evaluate(test_data[2])} ")
print(f"case 4: {test_data[3]} evaluates to {poker.evaluate(test_data[3])}")
print(f"case 5: {test_data[4]} evaluates to {poker.evaluate(test_data[4])}")
print(f"case 6: {test_data[5]} evaluates to {poker.evaluate(test_data[5])}")
print(f"case 7: {test_data[6]} evaluates to {poker.evaluate(test_data[6])}")
print(f"case 8: {test_data[7]} evaluates to {poker.evaluate(test_data[7])}")
print(f"case 9: {test_data[8]} evaluates to {poker.evaluate(test_data[8])}")
print(f"case 10: {test_data[9]} evaluates to {poker.evaluate(test_data[9])}")
