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
import mdtex2html

from multi_turn_alpaca.utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

load_8bit: bool = False
base_model: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "./multi-turn-alpaca"
prompt_template: str = "multi_turn"  # The prompt template to use, will default to alpaca.


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

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history):
    current_utterance = 'user: {instruction} assistant: '.format(instruction=input)
    if history:
        instruction = history + ' ' + current_utterance
    else:
        instruction = current_utterance
    prompt = prompter.generate_prompt(instruction)
    print('prompt: ' + prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
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
    response = output[len(prompt):]

    chatbot.append((parse_text(input), parse_text(response)))
    history = instruction + response
    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], ''


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Multi-turn Alpaca</h1>""")
    gr.HTML("""<h1 align="center">GitHub: https://github.com/l294265421/multi-turn-alpaca</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_tokens = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum Token Number", interactive=True)
            num_beams = gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            )
            do_sample = gr.components.Checkbox(value=False, label='Do sample')
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)

    history = gr.State('')
    # input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history
    submitBtn.click(predict, [user_input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history],
                    [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


server_name = '0.0.0.0'
share_gradio = True
demo.queue().launch(server_name=server_name, share=share_gradio, inbrowser=True)