import os

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


if __name__ == '__main__':
    task_data_dir = os.path.join(common_path.data_dir, 'training_data')
    # filenames = os.listdir(task_data_dir)
    filenames = ['alpaca_data_cleaned.txt', 'chat_alpaca.txt']
    output_lines = []
    for filename in filenames:
        if filename == 'training_data.txt':
            continue
        filepath = os.path.join(task_data_dir, filename)
        data_part = file_utils.read_all_lines(filepath)
        output_lines.extend(data_part)

    output_dir = os.path.join(common_path.data_dir, 'training_data')
    output_filepath = os.path.join(output_dir, 'training_data.txt')
    file_utils.write_lines(output_lines, output_filepath)
