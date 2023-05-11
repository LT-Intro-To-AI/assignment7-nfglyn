from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[10])

    if out == 0:
        #nothing in hand
        #convert into 4 digit table so no repeats
        output = [0,0,0,0]
    if out == 1:
        #pair
        #convert into 4 digit table so no repeats
        output = [1,0,0,0]
    if out == 2:
        #2 pair
        #convert into 4 digit table so no repeats
        output = [0,1,0,0]
    if out == 3:
        #3 of a kind
        #convert into 4 digit table so no repeats
        output = [0,0,1,0]
    if out == 4:
        #straight
        output = [1,1,0,0]
    if out == 5:
        #flush
        #convert into 4 digit table so no repeats
        output = [0,0,0,1]
    if out == 6:
        #full house
        #convert into 4 digit table so no repeats
        output = [1,0,1,0]
    if out == 7:
        #four of a kind
        #convert into 4 digit table so no repeats
        output = [1,1,1,0]
    if out == 8:
        #straight flush
        #convert into 4 digit table so no repeats
        output = [0,1,0,1]
    if out == 9:
        #royal flush
        #convert into 4 digit table so no repeats
        output = [1,0,0,1]
        
    inpt = [float(x) for x in tokens[:10]]
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

poker = NeuralNet(10, 5, 4)

# poker_hard_training = [
#     ([.1, 1, .1, .11, .1, .13, .1, .12, .1, .1], [.9]),
#     ([.2, .11, .2, .13, .2, 1, .2, .12, .2, .1], [.9]),
#     ([.3, .12, .3, .11, .3, .13, .3, 1, .3, .1], [.9]),
#     ([.4, 1, .4, .11, .4, .1, .4, .13, .4, .12], [.9]),
#     ([.4, .1, .4, .13, .4, .12, .4, .11, .4, 1], [.9]),
#     ([.1, .2, .1, .4, .1, .5, .1, .3, .1, .6], [.8]),
#     ([.1, .9, .1, .12, .1, 1, .1, .11, .1, .13], [.8]),
#     ([.2, .1, .2, .2, .2, .3, .2, .4, .2, .5], [.8]),
#     ([.3, .5, .3, .6, .3, .9, .3, .7, .3, .8], [.8]),
#     ([.4, .1, .4, .4, .4, .2, .4, .3, .4, .5], [.8]),
#     ([.1, .1, .2, .1, .3, .9, .1, .5, .2, .3], [.1]),
#     ([.2, .6, .2, .1, .4, .13, .2, .4, .4, .9], [0]),
#     ([.1, 1, .4, .6, .1, .2, .1, .1, .3, .8], [0]),
#     ([.2, .13, .2, .1, .4, .4, .1, .5, .2, .11], [0]),
#     ([.3, .8, .4, 1.2, .3, .9, .4, .2, .3, .2], [.1])

    # ([.1, 1, .1, 1.1, .1, 1.3, .1, 1.2, .1, .1], [.9]),
    # ([.2, 1.1, .2, 1.3, .2, 1, .2, 1.2, .2, .1], [.9]),
    # ([.3, 1.2, .3, 1.1, .3, 1.3, .3, 1, .3, .1], [.9]),
    # ([.4, 1, .4, 1.1, .4, .1, .4, 1.3, .4, 1.2], [.9]),
    # ([.4, .1, .4, 1.3, .4, 1.2, .4, 1.1, .4, 1], [.9]),
    # ([.1, .2, .1, .4, .1, .5, .1, .3, .1, .6], [.8]),
    # ([.1, .9, .1, 1.2, .1, 1.0, .1, 1.1, .1, 1.3], [.8]),
    # ([.2, .1, .2, .2, .2, .3, .2, .4, .2, .5], [.8]),
    # ([.3, .5, .3, .6, .3, .9, .3, .7, .3, .8], [.8]),
    # ([.4, .1, .4, .4, .4, .2, .4, .3, .4, .5], [.8]),
    # ([.1, .1, .2, .1, .3, .9, .1, .5, .2, .3], [.1]),
    # ([.2, .6, .2, .1, .4, 1.3, .2, .4, .4, .9], [0]),
    # ([.1, 1, .4, .6, .1, .2, .1, .1, .3, .8], [0]),
    # ([.2, 1.3, .2, .1, .4, .4, .1, .5, .2, 1.1], [0]),
    # ([.3, .8, .4, 1.2, .3, .9, .4, .2, .3, .2], [.1])
# ]

# test_data_0 = [
# [.1, .1, .1, 1.3, .2, .4, .2, .3, .1, 1.2],
# [.3, 1.2, .3, .2, .3, 1.1, .4, .5, .2, .5],
# [.1, .9, .4, .6, .1, .4, .3, .2, .3, .9],
# [.1, .4, .3, 1.3, .2, 1.3, .2, .1, .3, .6],
# [.3, 1, .2, .7, .1, .2, .2, 1.1, .4, .9],
# [.1, .3, .4, .5, .3, .4, .1, 1.2, .4, .6],
# [.2, .6, .4, 1.1, .2, .3, .4, .9, .1, .7],
# [.3, .2, .4, .9, .3, .7, .4, .3, .4, .5],
# [.4, .4, .3, 1.3, .1, .8, .3, .9, .3, 1],
# [.1, .9, .3, .8, .4, .4, .1, .7, .3, .5],
# ]

# test_data_1 = [
#     [.2, .5, .3, .13, .3, .12, .3, .7, .1, .13],
#     [ .4, .8, .3, .2, .2, .8, .2, .9, .3, .11],
#     [ .1, .8, .1, .12, .2, .9, .2, .6, .3, .2],
#     [.1, .5, .3, .13, .2, .13, .2, .7, .4, .5],
#     [.4, .7, .4, 1, .3, .1, .3, .3, .2, .5],
#     [.3, .13, .2, .2, .4, 1, .3, .3, .2, .13],
#     [.1, .5, .3, .7, .1, .3, .1, .6, .2, .8],
#     [.1, .2, .4, .3, .2, .8, .4, .11, .4, .13],
#     [.2, .1, .4, .4, .1, .4, .11, .3, .7],
#     [.1, .4, .4, .1, .4, .12, .2, .3, .4, .13],
# ]

# test_data = normalize("poker-hand-testing.data", "r")
# for test
# test_data 
with open("poker-hand-training-true.data", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4] 

# for line in training_data:
#     print(line)

poker_hand_training = normalize(training_data)

# for line in poker_hand_training:
#     print(line)

# train_data, test_data = train_test_split(poker_hand_training, test_size=0.1, train_size= 0.1, random_state= 3)
# poker.train(train_data)
# print(test_data)
with open("poker-hand-testing.data", "r") as f:
    testing_data = [parse_line(line) for line in f.readlines() if len(line) > 4] 
test_data = normalize(testing_data)

poker.train(poker_hand_training)

for i in poker.test_with_expected(test_data):
    print(f"desired: {i[1]}, actual: {i[2]}")


# print(f"case 1: {test_data[0]} evaluates to {poker.evaluate(test_data[0])} actual result: {test_data[0][10]} ")
# print(f"case 2: {test_data[1]} evaluates to {poker.evaluate(test_data[1])} actual result: {test_data[1][10]} ")
# print(f"case 3: {test_data[2]} evaluates to {poker.evaluate(test_data[2])} actual result: {test_data[2][10]} ")
# print(f"case 4: {test_data[3]} evaluates to {poker.evaluate(test_data[3])} actual result: {test_data[3][10]} ")
# print(f"case 5: {test_data[4]} evaluates to {poker.evaluate(test_data[4])} actual result: {test_data[4][10]} ")
# print(f"case 6: {test_data[5]} evaluates to {poker.evaluate(test_data[5])} actual result: {test_data[5][10]} ")
# print(f"case 7: {test_data[6]} evaluates to {poker.evaluate(test_data[6])} actual result: {test_data[6][10]} ")
# print(f"case 8: {test_data[7]} evaluates to {poker.evaluate(test_data[7])} actual result: {test_data[7][10]} ")
# print(f"case 9: {test_data[8]} evaluates to {poker.evaluate(test_data[8])} actual result: {test_data[8][10]} ")
# print(f"case 10: {test_data[9]} evaluates to {poker.evaluate(test_data[9])} actual result: {test_data[9][10]} ")
