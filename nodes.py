from .nai import NovelAIAPI
from .config import NAI_API_KEY
class NovelAINode:
    CATEGORY = "NovelAI"
    samplers = ["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde", "k_dpmpp_2m", "k_dpmpp_sde", "ddim_v3"]
    schedulers = ["native", "karras", "exponential", "polyexponential"]
    models = ["","nai-diffusion-2", "nai-diffusion-3", "nai-diffusion-furry-3"]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_positive": ("STRING", {"default": "CLIP_POSITIVE","multiline": True}),
                "prompt_negative": ("STRING", {"default": "CLIP_NEGATIVE","multiline": True}),
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

    def generate_image(self, input_image=None, prompt_positive="", prompt_negative="", noise_seed=0,
                       sampler="k_euler_ancestral",model="nai-diffusion-3", scheduler="karras", smea="enable", smea_dyn="enable", steps=28,
                       cfg_scale=6.0, width=832, height=1216):
        print("into")
        if(NAI_API_KEY == "<KEY>"):
            raise Exception("API key not set,please configure your API key in config.py in the plugin directory.")
        api = NovelAIAPI(api_key=NAI_API_KEY)
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
            reference_image_multiple=[],
            reference_information_extracted_multiple=[],
            reference_strength_multiple=[],
            input_image=input_image
        ))
class NovelAIVibe:
    CATEGORY = "NovelAI_VIBE"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "2_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "2_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "3_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "3_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
                "4_information_extracted": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0}),
                "4_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),

            },
            "optional": {
                "input_image_2": ("IMAGE",),  # Optional image input
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("VIBE_PIPE",)
    FUNCTION = "process_vibe"

    def process_vibe(self, ):
        return ()