# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from typing import Optional
from cog import BasePredictor, Input, Path, BaseModel
import torch
from PIL import Image
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
from deepseek_vl2.serve.inference import (
    convert_conversation_to_prompts,
    load_model,
)
from web_demo import generate_prompt_with_history


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/deepseek-ai/deepseek-vl2-small/model_cache.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class ModelOutput(BaseModel):
    img_out: Optional[Path]
    text_out: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        self.dtype = torch.bfloat16
        self.tokenizer, self.vl_gpt, self.vl_chat_processor = load_model(
            f"{MODEL_CACHE}/deepseek-ai/deepseek-vl2-small", dtype=self.dtype
        )

    def predict(
        self,
        text: str = Input(
            description="Input text.",
            default="Describe this image.",
        ),
        image1: Path = Input(description="First image"),
        image2: Path = Input(
            description="Optional, second image for multiple images image2text",
            default=None,
        ),
        image3: Path = Input(
            description="Optional, third image for multiple images image2text",
            default=None,
        ),
        max_new_tokens: int = Input(
            description="The maximum numbers of tokens to generate",
            le=4096,
            ge=0,
            default=2048,
        ),
        temperature: float = Input(
            description="The value used to modulate the probabilities of the next token. Set the temperature to 0 for deterministic generation",
            default=0.1,
        ),
        top_p: float = Input(
            description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
            default=0.9,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty", le=2, ge=0, default=1.1
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        pil_images = [
            Image.open(str(img)).convert("RGB")
            for img in [image1, image2, image3]
            if img
        ]

        conversation = generate_prompt_with_history(
            text,
            pil_images,
            None,
            self.vl_chat_processor,
            self.tokenizer,
            max_length=4096,
        )

        all_conv, _ = convert_conversation_to_prompts(conversation)
        print(all_conv)

        prepare_inputs = self.vl_chat_processor(
            conversations=all_conv,
            images=pil_images,
            force_batchify=True,
        ).to(self.vl_gpt.device, dtype=self.dtype)

        with torch.no_grad():
            inputs_embeds, past_key_values = self.vl_gpt.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
            )

            outputs = self.vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            answer = self.tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]) :].cpu().tolist(),
                skip_special_tokens=False,
            )
            vg_image = parse_ref_bbox(answer, image=pil_images[-1])

            out_img = "out.png"
            if vg_image is not None:
                vg_image.save(out_img, format="JPEG", quality=85)

        return ModelOutput(
            text_out=answer, img_out=Path(out_img) if vg_image is not None else None
        )
