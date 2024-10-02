import base64
import io
import re

from PIL import Image

from .nai import NovelAIAPI
from .config import NAI_API_KEY


class NovelAINode:
    CATEGORY = "NovelAI"
    samplers = ["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde", "k_dpmpp_2m", "k_dpmpp_sde",
                "ddim_v3"]
    schedulers = ["native", "karras", "exponential", "polyexponential"]
    models = ["", "nai-diffusion-2", "nai-diffusion-3", "nai-diffusion-furry-3"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_positive": ("STRING", {"default": "POSITIVE prompt", "multiline": True}),
                "prompt_negative": ("STRING", {"default": "NEGATIVE prompt", "multiline": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": (cls.samplers, {"default": "k_euler_ancestral"}),  # Choose sampler
                "model": (cls.models, {"default": "nai-diffusion-3"}),
                "scheduler": (cls.schedulers, {"default": "karras"}),  # Choose scheduler
                "smea": (["enable", "disable"], {"default": "enable"}),  # SMEA setting
                "smea_dyn": (["enable", "disable"], {"default": "enable"}),  # Dynamic SMEA setting
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2560}),  # Image width
                "height": ("INT", {"default": 1216, "min": 64, "max": 2560}),  # Image height
            },
            "optional": {
                "input_image": ("IMAGE",),  # Optional image input
                "vibe_pipe": ("VIBE_PIPE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"

    def generate_image(self, input_image=None, vibe_pipe=None, prompt_positive="", prompt_negative="", noise_seed=0,
                       sampler="k_euler_ancestral", model="nai-diffusion-3", scheduler="karras", smea="enable",
                       smea_dyn="enable", steps=28,
                       cfg_scale=6.0, width=832, height=1216):
        print("into")
        if (NAI_API_KEY == "<KEY>"):
            raise Exception("API key not set,please configure your API key in config.py in the plugin directory.")
        api = NovelAIAPI(api_key=NAI_API_KEY)
        reference_image_multiple = []
        reference_information_extracted_multiple = []
        reference_strength_multiple = []
        if vibe_pipe is not None:
            reference_image_multiple = vibe_pipe["reference_image_multiple"]
            reference_information_extracted_multiple = vibe_pipe["reference_information_extracted_multiple"]
            reference_strength_multiple = vibe_pipe["reference_strength_multiple"]
        return (api.generate_image(
            input_text=prompt_positive,
            model=model,  # Assuming 'novelai' model for now
            width=width,
            height=height,
            scale=cfg_scale,
            sampler=sampler,
            steps=steps,
            n_samples=1,
            ucPreset=0,
            qualityToggle=True,
            sm=(smea == "enable"),
            sm_dyn=(smea_dyn == "enable"),
            dynamic_thresholding=False,
            controlnet_strength=1,
            legacy=False,
            add_original_image=True,
            cfg_rescale=0,
            noise_schedule=scheduler,
            legacy_v3_extend=False,
            seed=noise_seed if noise_seed else None,
            negative_prompt=prompt_negative,
            reference_image_multiple=reference_image_multiple,
            reference_information_extracted_multiple=reference_information_extracted_multiple,
            reference_strength_multiple=reference_strength_multiple,
            input_image=input_image
        ))

class NovelAIPrompt:
    CATEGORY = "NovelAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "prompt", "multiline": True}),
                "To": (["NovelAI", "Comfyui"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_prompt"

    def process_prompt(self, prompt=None, To=None):
        def custom_round(value, decimals=4):
            factor = 10 ** decimals
            return int(value * factor) / factor
        if To == "NovelAI":
            # Convert increasing weights from () to {}
            def convert_increasing_weight(match):
                word = match.group(1)
                weight = float(match.group(2))
                brace_count = int(round((weight - 1) / 0.05))  # Calculate curly brace count based on weight
                return f"{'{' * brace_count}{word.replace('_', ' ')}{'}' * brace_count}"

                # Convert decreasing weights from () to []

            def convert_decreasing_weight(match):
                word = match.group(1)
                weight = float(match.group(2))
                bracket_count = int(round((1 - weight) / 0.05))  # Calculate square bracket count based on weight
                return f"{'[' * bracket_count}{word.replace('_', ' ')}{']' * bracket_count}"

                # Replace increasing weights with curly braces and decreasing weights with square brackets

            prompt = re.sub(r'\((\w[\w\s()]+):([1-9]\.\d+)\)', convert_increasing_weight, prompt)
            prompt = re.sub(r'\((\w[\w\s()]+):([0]\.\d+)\)', convert_decreasing_weight, prompt)

            # Ensure parts of the prompt are comma-separated
            prompt = re.sub(r'\s*,\s*', ',', prompt)  # Replace all spaces around commas with clean commas
            return (prompt,)
        elif To == "Comfyui":
            re_attention = re.compile(r'\{|\[|\}|\]|[^\{\}\[\]]+')
            text = prompt.replace("(", "\\(").replace(")", "\\)")
            text = re.sub(r'\\{2,}(\(|\))', r'\\\1', text)

            res = []
            curly_brackets = []
            square_brackets = []

            curly_bracket_multiplier = 1.05
            square_bracket_multiplier = 1 / 1.05

            def multiply_range(start_position, multiplier):
                for pos in range(start_position, len(res)):
                    res[pos][1] = custom_round(res[pos][1] * multiplier)

            for match in re_attention.finditer(text):
                word = match.group(0)

                if word == "{":
                    curly_brackets.append(len(res))
                elif word == "[":
                    square_brackets.append(len(res))
                elif word == "}" and curly_brackets:
                    multiply_range(curly_brackets.pop(), curly_bracket_multiplier)
                elif word == "]" and square_brackets:
                    multiply_range(square_brackets.pop(), square_bracket_multiplier)
                else:
                    res.append([word, 1.0])

            for pos in curly_brackets:
                multiply_range(pos, curly_bracket_multiplier)

            for pos in square_brackets:
                multiply_range(pos, square_bracket_multiplier)

            if not res:
                res = [["", 1.0]]

            # Merge runs of identical weights
            i = 0
            while i + 1 < len(res):
                if res[i][1] == res[i + 1][1]:
                    res[i][0] += res[i + 1][0]
                    res.pop(i + 1)
                else:
                    i += 1

            result = ""
            for item in res:
                if item[1] == 1.0:
                    result += item[0]
                else:
                    result += f"({item[0]}:{item[1]})"

            return (result,)


class NovelAIVibe:
    CATEGORY = "NovelAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "image2_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "image2_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "image3_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "image3_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "image4_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "image4_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),

            },
            "optional": {
                "input_image_2": ("IMAGE",),  # Optional image input
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "pre_vibe_pipe": ("VIBE_PIPE",)
            }
        }

    RETURN_TYPES = ("VIBE_PIPE",)
    FUNCTION = "process_vibe"

    def toBase64(self, image):
        image = image.squeeze(0)
        image = (image * 255).byte().numpy()
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return image_payload

    def process_vibe(self, input_image=None, pre_vibe_pipe=None, input_image_2=None, input_image_3=None,
                     input_image_4=None, information_extracted=1.0,
                     image2_information_extracted=1.0, image3_information_extracted=1.0,
                     image4_information_extracted=1.0, strength=0.7,
                     image2_strength=0.7, image3_strength=0.7, image4_strength=0.7):
        vibe_pipe = {
            "reference_image_multiple": [],
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": []
        }
        if pre_vibe_pipe is not None:
            vibe_pipe["reference_image_multiple"] += (pre_vibe_pipe["reference_image_multiple"])
            vibe_pipe["reference_strength_multiple"] += (pre_vibe_pipe["reference_strength_multiple"])
            vibe_pipe["reference_information_extracted_multiple"] += (
                pre_vibe_pipe["reference_information_extracted_multiple"])
        images = [[input_image, information_extracted, strength],
                  [input_image_2, image2_information_extracted, image2_strength],
                  [input_image_3, image3_information_extracted, image3_strength],
                  [input_image_4, image4_information_extracted, image4_strength]]
        for image in images:
            if image[0] is not None:
                vibe_pipe["reference_image_multiple"] += [self.toBase64(image[0])]
                vibe_pipe["reference_information_extracted_multiple"] += [image[1]]
                vibe_pipe["reference_strength_multiple"] += [image[2]]
        return (vibe_pipe,)


class NovelAILineart:
    CATEGORY = "NovelAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tolineart"

    def tolineart(self, input_image=None):
        if (NAI_API_KEY == "<KEY>"):
            raise Exception("API key not set,please configure your API key in config.py in the plugin directory.")
        api = NovelAIAPI(api_key=NAI_API_KEY)
        return (api.toLineArt(input_image))


class NovelAISketch:
    CATEGORY = "NovelAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tosketch"

    def tosketch(self, input_image=None):
        if (NAI_API_KEY == "<KEY>"):
            raise Exception("API key not set,please configure your API key in config.py in the plugin directory.")
        api = NovelAIAPI(api_key=NAI_API_KEY)
        return (api.toSketchArt(input_image))


class NovelAIDeclutter:
    CATEGORY = "NovelAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "declutter"

    def declutter(self, input_image=None):
        if (NAI_API_KEY == "<KEY>"):
            raise Exception("API key not set,please configure your API key in config.py in the plugin directory.")
        api = NovelAIAPI(api_key=NAI_API_KEY)
        return (api.declutter(input_image))
