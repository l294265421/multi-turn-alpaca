import os
import json

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


if __name__ == '__main__':
    filepath = os.path.join(common_path.project_dir, 'data/original_data/ChatAlpaca/chatalpaca_data_10k.json')
    data = file_utils.load_json(filepath)
    output_lines = []
    for i, instance in enumerate(data):
        temp = ['%s: %s' % (e['role'], e['content']) for e in instance]
        last_utterance_parts = temp[-1].split(' ')
        history = ' '.join(temp[: len(temp) - 1]) + ' ' + last_utterance_parts[0]
        output = instance[-1]['content']
        output_lines.append(json.dumps({'instruction': history, 'input': '', 'output': output}))

    output_dir = os.path.join(common_path.data_dir, 'training_data')
    output_filepath = os.path.join(output_dir, 'chat_alpaca.txt')
    file_utils.write_lines(output_lines, output_filepath)
