# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*- coding:utf-8 -*-
from argparse import ArgumentParser

import io
import sys
import base64
from PIL import Image

import gradio as gr
import torch

from deepseek_vl2.serve.app_modules.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    wrap_gen_fn,
)
from deepseek_vl2.serve.app_modules.overwrites import reload_javascript
from deepseek_vl2.serve.app_modules.presets import (
    CONCURRENT_COUNT,
    MAX_EVENTS,
    description,
    description_top,
    title
)
from deepseek_vl2.serve.app_modules.utils import (
    configure_logger,
    is_variable_assigned,
    strip_stop_words,
    parse_ref_bbox,
    pil_to_base64,
    display_example
)

from deepseek_vl2.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from deepseek_vl2.models.conversation import SeparatorStyle

logger = configure_logger()

MODELS = [
    "DeepSeek-VL2-tiny",
    "DeepSeek-VL2-small",
    "DeepSeek-VL2",

    "deepseek-ai/deepseek-vl2-tiny",
    "deepseek-ai/deepseek-vl2-small",
    "deepseek-ai/deepseek-vl2",
]

DEPLOY_MODELS = dict()
IMAGE_TOKEN = "<image>"

examples_list = [
    # visual grounding - 1
    [
        ["images/visual_grounding_1.jpeg"],
        "<|ref|>The giraffe at the back.<|/ref|>",
    ],

    # visual grounding - 2
    [
        ["images/visual_grounding_2.jpg"],
        "找到<|ref|>淡定姐<|/ref|>",
    ],

    # visual grounding - 3
    [
        ["images/visual_grounding_3.png"],
        "Find all the <|ref|>Watermelon slices<|/ref|>",
    ],

    # grounding conversation
    [
        ["images/grounding_conversation_1.jpeg"],
        "<|grounding|>I want to throw out the trash now, what should I do?",
    ],

    # in-context visual grounding
    [
        [
            "images/incontext_visual_grounding_1.jpeg",
            "images/icl_vg_2.jpeg"
        ],
        "<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image."
    ],

    # vqa
    [
        ["images/vqa_1.jpg"],
        "Describe each stage of this image in detail",
    ],

    # multi-images
    [
        [
            "images/multi_image_1.jpeg",
            "images/multi_image_2.jpeg",
            "images/multi_image_3.jpeg"
        ],
        "能帮我用这几个食材做一道菜吗?",
    ]

]


def fetch_model(model_name: str, dtype=torch.bfloat16):
    global args, DEPLOY_MODELS

    if args.local_path:
        model_path = args.local_path
    else:
        model_path = model_name

    if model_name in DEPLOY_MODELS:
        model_info = DEPLOY_MODELS[model_name]
        print(f"{model_name} has been loaded.")
    else:
        print(f"{model_name} is loading...")
        DEPLOY_MODELS[model_name] = load_model(model_path, dtype=dtype)
        print(f"Load {model_name} successfully...")
        model_info = DEPLOY_MODELS[model_name]

    return model_info


def generate_prompt_with_history(
    text, images, history, vl_chat_processor, tokenizer, max_length=2048
):
    """
    Generate a prompt with history for the deepseek application.

    Args:
        text (str): The text prompt.
        images (list[PIL.Image.Image]): The image prompt.
        history (list): List of previous conversation messages.
        tokenizer: The tokenizer used for encoding the prompt.
        max_length (int): The maximum length of the prompt.

    Returns:
        tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
    """
    global IMAGE_TOKEN

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if images is not None and len(images) > 0:

        num_image_tags = text.count(IMAGE_TOKEN)
        num_images = len(images)

        if num_images > num_image_tags:
            pad_image_tags = num_images - num_image_tags
            image_tokens = "\n".join([IMAGE_TOKEN] * pad_image_tags)

            # append the <image> in a new line after the text prompt
            text = image_tokens + "\n" + text
        elif num_images < num_image_tags:
            remove_image_tags = num_image_tags - num_images
            text = text.replace(IMAGE_TOKEN, "", remove_image_tags)

        # print(f"prompt = {text}, len(images) = {len(images)}")
        text = (text, images)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()
    logger.info("=" * 80)
    logger.info(get_prompt(conversation))

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = (
            current_prompt.replace("</s>", "")
            if sft_format == "deepseek"
            else current_prompt
        )

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            gr.Error("The messages between user and assistant are not paired.")
            return

        try:
            for _ in range(2):  # pop out two messages in a row
                conversation.messages.pop(0)
        except IndexError:
            gr.Error("Input text processing failed, unable to respond in this round.")
            return None

    gr.Error("Prompt could not be generated within max_length limit.")
    return None


def to_gradio_chatbot(conv):
    """Convert the conversation to gradio chatbot format."""
    ret = []
    for i, (role, msg) in enumerate(conv.messages[conv.offset:]):
        if i % 2 == 0:
            if type(msg) is tuple:
                msg, images = msg

                if isinstance(images, list):
                    for j, image in enumerate(images):
                        if isinstance(image, str):
                            with open(image, "rb") as f:
                                data = f.read()
                            img_b64_str = base64.b64encode(data).decode()
                            image_str = (f'<img src="data:image/png;base64,{img_b64_str}" '
                                         f'alt="user upload image" style="max-width: 300px; height: auto;" />')
                        else:
                            image_str = pil_to_base64(image, f"user upload image_{j}", max_size=800, min_size=400)

                        # replace the <image> tag in the message
                        msg = msg.replace(IMAGE_TOKEN, image_str, 1)

                else:
                    pass

            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def to_gradio_history(conv):
    """Convert the conversation to gradio history state."""
    return conv.messages[conv.offset:]


def get_prompt(conv) -> str:
    """Get the prompt for generation."""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:  # multimodal message
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt()


def transfer_input(input_text, input_images):
    print("transferring input text and input image")

    return (
        input_text,
        input_images,
        gr.update(value=""),
        gr.update(value=None),
        gr.Button(visible=True)
    )


@wrap_gen_fn
def predict(
    text,
    images,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    """
    Function to predict the response based on the user's input and selected model.

    Parameters:
    user_text (str): The input text from the user.
    user_image (str): The input image from the user.
    chatbot (str): The chatbot's name.
    history (str): The history of the chat.
    top_p (float): The top-p parameter for the model.
    temperature (float): The temperature parameter for the model.
    max_length_tokens (int): The maximum length of tokens for the model.
    max_context_length_tokens (int): The maximum length of context tokens for the model.
    model_select_dropdown (str): The selected model from the dropdown.

    Returns:
    generator: A generator that yields the chatbot outputs, history, and status.
    """
    print("running the prediction function")
    try:
        tokenizer, vl_gpt, vl_chat_processor = fetch_model(model_select_dropdown)

        if text == "":
            yield chatbot, history, "Empty context."
            return
    except KeyError:
        yield [[text, "No Model Found"]], [], "No Model Found"
        return

    if images is None:
        images = []

    # load images
    pil_images = []
    for img_or_file in images:
        try:
            # load as pil image
            if isinstance(images, Image.Image):
                pil_images.append(img_or_file)
            else:
                image = Image.open(img_or_file.name).convert("RGB")
                pil_images.append(image)
        except Exception as e:
            print(f"Error loading image: {e}")

    conversation = generate_prompt_with_history(
        text,
        pil_images,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
    )
    all_conv, last_image = convert_conversation_to_prompts(conversation)

    stop_words = conversation.stop_str
    gradio_chatbot_output = to_gradio_chatbot(conversation)

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            conversations=all_conv,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            chunk_size=args.chunk_size
        ):
            full_response += x
            response = strip_stop_words(full_response, stop_words)
            conversation.update_last_message(response)
            gradio_chatbot_output[-1][1] = response

            # sys.stdout.write(x)
            # sys.stdout.flush()

            yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

    if last_image is not None:
        # TODO always render the last image's visual grounding image
        vg_image = parse_ref_bbox(response, last_image)
        if vg_image is not None:
            vg_base64 = pil_to_base64(vg_image, f"vg", max_size=800, min_size=400)
            gradio_chatbot_output[-1][1] += vg_base64
            yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

    print("flushed result to gradio")
    torch.cuda.empty_cache()

    if is_variable_assigned("x"):
        print(f"{model_select_dropdown}:\n{text}\n{'-' * 80}\n{x}\n{'=' * 80}")
        print(
            f"temperature: {temperature}, "
            f"top_p: {top_p}, "
            f"repetition_penalty: {repetition_penalty}, "
            f"max_length_tokens: {max_length_tokens}"
        )

    yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"


# @wrap_gen_fn
def retry(
    text,
    images,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    if len(history) == 0:
        yield (chatbot, history, "Empty context")
        return

    chatbot.pop()
    history.pop()
    text = history.pop()[-1]
    if type(text) is tuple:
        text, image = text

    yield from predict(
        text,
        images,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
        model_select_dropdown,
        args.chunk_size
    )


def preview_images(files):
    if files is None:
        return []

    image_paths = []
    for file in files:
        # 使用 file.name 获取文件路径
        # image = Image.open(file.name)
        image_paths.append(file.name)
    return image_paths  # 返回所有图片路径，用于预览


def build_demo(args):
    # fetch model
    if not args.lazy_load:
        fetch_model(args.model_name)

    with open("deepseek_vl2/serve/assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_images = gr.State()

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="deepseek_chatbot",
                        show_share_button=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(
                            show_label=False, placeholder="Enter text", container=False
                        )
                    with gr.Column(
                        min_width=70,
                    ):
                        submitBtn = gr.Button("Send")
                    with gr.Column(
                        min_width=70,
                    ):
                        cancelBtn = gr.Button("Stop")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                upload_images = gr.Files(file_types=["image"], show_label=True)
                gallery = gr.Gallery(columns=[3], height="200px", show_label=True)

                upload_images.change(preview_images, inputs=upload_images, outputs=gallery)

                with gr.Tab(label="Parameter Setting") as parameter_row:
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=4096,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
                    model_select_dropdown = gr.Dropdown(
                        label="Select Models",
                        choices=[args.model_name],
                        multiselect=False,
                        value=args.model_name,
                        interactive=True,
                    )

                    # show images, but not visible
                    show_images = gr.HTML(visible=False)
                    # show_images = gr.Image(type="pil", interactive=False, visible=False)

        def format_examples(examples_list):
            examples = []
            for images, texts in examples_list:
                examples.append([images, display_example(images), texts])

            return examples

        gr.Examples(
            examples=format_examples(examples_list),
            inputs=[upload_images, show_images, text_box],
        )

        gr.Markdown(description)

        input_widgets = [
            input_text,
            input_images,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
            model_select_dropdown,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, upload_images],
            outputs=[input_text, input_images, text_box, upload_images, submitBtn],
            show_progress=True,
        )

        predict_args = dict(
            fn=predict,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        retry_args = dict(
            fn=retry,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[text_box, status_display]
        )

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submitBtn.click(**transfer_input_args).then(**predict_args),
        ]

        emptyBtn.click(reset_state, outputs=output_widgets, show_progress=True)
        emptyBtn.click(**reset_args)
        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            output_widgets,
            show_progress=True,
        )

        cancelBtn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    return demo


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=MODELS, help="model name")
    parser.add_argument("--local_path", type=str, default="", help="huggingface ckpt, optional")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="ip address")
    parser.add_argument("--port", type=int, default=37913, help="port number")
    parser.add_argument("--root_path", type=str, default="", help="root path")
    parser.add_argument("--lazy_load", action='store_true')
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()

    demo = build_demo(args)
    demo.title = "DeepSeek-VL2 Chatbot"

    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT, max_size=MAX_EVENTS).launch(
        # share=False,
        share=True,
        favicon_path="deepseek_vl2/serve/assets/favicon.ico",
        inbrowser=False,
        server_name=args.ip,
        server_port=args.port,
        root_path=args.root_path
    )
