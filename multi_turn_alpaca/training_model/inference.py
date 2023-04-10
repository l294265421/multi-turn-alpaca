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


def add_line_separators(history):
    """

    :param history:
    :return:
    """
    return history


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "./multi-turn-alpaca",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
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
        input=None,
        max_new_tokens=512,
        num_beams=4,
        do_sample=False,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        is_clean_history=False,
        user_id=0,
        **kwargs,
    ):
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


        prompt = prompter.generate_prompt(instruction, input)
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
        user_history[user_id] = instruction + prompter.get_response(output)
        return add_line_separators(instruction), prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
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
            gr.components.Checkbox(value=False, label='Clean history'),
            gr.components.Slider(
                minimum=0, maximum=40000000, step=1, value=random.randint(0, 40000000), label="User ID"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="History",
            ),
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Multi-turn Alpaca",
        description="Multi-turn Alpaca",  # noqa: E501
    ).launch(server_name=server_name, share=share_gradio)
    # Old testing code follows.


if __name__ == "__main__":
    fire.Fire(main)
