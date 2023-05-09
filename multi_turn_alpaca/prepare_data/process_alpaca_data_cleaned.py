import os
import json

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


if __name__ == '__main__':
    filepath = os.path.join(common_path.project_dir, 'data/original_data/AlpacaDataCleaned/alpaca_data_cleaned.json')
    data = file_utils.load_json(filepath)
    output_lines = []
    for i, instance in enumerate(data):
        instruction = "user: {instruction} {input} assistant: ".format_map(instance)
        instance['instruction'] = instruction
        instance['input'] = ''
        output_lines.append(json.dumps(instance))

    output_dir = os.path.join(common_path.data_dir, 'training_data')
    output_filepath = os.path.join(output_dir, 'alpaca_data_cleaned.txt')
    file_utils.write_lines(output_lines, output_filepath)
