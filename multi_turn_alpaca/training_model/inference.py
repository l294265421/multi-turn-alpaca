import os
import sys
import random

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

from multi_turn_alpaca.utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# key: user id, value: conversation history
user_history = {}


def add_line_separators(history: str):
    """

    :param history:
    :return:
    """
    if not history:
        return ''
    utterances = []
    user = 'user: '
    assistant = 'assistant: '
    current_role = assistant
    end = history.find(current_role)
    while end != -1:
        utterance = history[: end]
        utterances.append(utterance)
        history = history[end:]
        if current_role == user:
            current_role = assistant
        else:
            current_role = user
        end = history.find(current_role)
    utterances.append(history)
    result = '\n'.join(utterances)
    return result


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "./multi-turn-alpaca",
    prompt_template: str = "multi_turn",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    prompter = Prompter(prompt_template)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        # input=None,
        max_new_tokens=512,
        num_beams=4,
        do_sample=False,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        user_id=0,
        is_clean_history=False,
        **kwargs,
    ):
        print('user id: %s' % user_id)
        current_utterance = 'user: {instruction} assistant: '.format(instruction=instruction)
        if is_clean_history:
            if user_id in user_history:
                user_history.pop(user_id)
            history = ''
        else:
            if user_id in user_history:
                history = user_history[user_id] + ' '
            else:
                history = ''
        instruction = history + current_utterance

        prompt = prompter.generate_prompt(instruction)
        print('prompt:' + prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            # top_k=top_k,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print('output: ' + output)
        response = output[len(prompt):]
        user_history[user_id] = instruction + response
        return add_line_separators(instruction), response

    # while True:
    #     print()
    #     instruction = input("Instruction: ")
    #     if instruction == 'quit':
    #         break
    #
    #     result = evaluate(instruction, max_new_tokens=8)
    #     print('history:')
    #     print(result[0])
    #     print('response:')
    #     print(result[1])

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            # gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Checkbox(value=False, label='Do sample'),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=0, maximum=40000000, step=1, value=random.randint(0, 40000000), label="User ID"
            ),
            gr.components.Checkbox(value=False, label='Clean history'),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=21,
                label="History",
            ),
            gr.inputs.Textbox(
                lines=3,
                label="Output",
            )
        ],
        title="Multi-turn Alpaca",
        description="GitHub: https://github.com/l294265421/multi-turn-alpaca",  # noqa: E501
    ).launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
