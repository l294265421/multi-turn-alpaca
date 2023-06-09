import os
import json

from multi_turn_alpaca.common import common_path
from multi_turn_alpaca.utils import file_utils


def generate_instruction(conversation):
    history = ['%s: %s' % (e['role'], e['content']) for e in conversation[: -1]]
    last_utterance_role = conversation[-1]['role']
    history = ' '.join(history) + ' ' + last_utterance_role + ': '
    output = conversation[-1]['content']
    instruction = json.dumps({'instruction': history, 'input': '', 'output': output})
    return instruction


if __name__ == '__main__':
    filepath = os.path.join(common_path.project_dir, 'data/original_data/ChatAlpaca/chatalpaca_data_10k.json')
    data = file_utils.load_json(filepath)
    output_lines = []
    for i, instance in enumerate(data):
        end = 2
        while end <= len(instance):
            conversation = instance[: end]
            instruction = generate_instruction(conversation)
            output_lines.append(instruction)
            end += 2

    output_dir = os.path.join(common_path.data_dir, 'training_data')
    output_filepath = os.path.join(output_dir, 'chat_alpaca.txt')
    file_utils.write_lines(output_lines, output_filepath)
