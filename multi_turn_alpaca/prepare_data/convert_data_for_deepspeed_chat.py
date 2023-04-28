import os
import random

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


if __name__ == '__main__':
    data_path = os.path.join(common_path.data_dir, 'training_data', 'training_data.txt')
    lines = file_utils.read_all_lines(data_path)
    train_data = []
    test_data = []
    test_size = 0.2
    for line in lines:
        converted_line = line.replace('user: ', 'Human: ').replace('assistant: ', 'Assistant: ')
        threshold = random.random()
        if threshold < test_size:
            test_data.append(converted_line)
        else:
            train_data.append(converted_line)
    file_utils.write_lines(train_data, data_path + '.train')
    file_utils.write_lines(test_data, data_path + '.test')
