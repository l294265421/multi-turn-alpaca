import os
import json

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


if __name__ == '__main__':
    filepath = os.path.join(common_path.project_dir, 'data/original_data/ChatAlpaca/chatalpaca_data_10k.json')
    data = file_utils.load_json(filepath)
    output_lines = []
    for i, instance in enumerate(data):
        history = ''
        for one_turn in instance:
            role = one_turn['role']
            content = one_turn['content']
            history += ('{role}: '.format(role=role))
            if role != 'user':
                output_lines.append(json.dumps({'instruction': history, 'input': '', 'output': content}))
            history += ('%s ' % content)

    output_dir = os.path.join(common_path.data_dir, 'training_data')
    output_filepath = os.path.join(output_dir, 'chat_alpaca.txt')
    file_utils.write_lines(output_lines, output_filepath)
