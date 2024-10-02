from .nodes import NovelAINode,NovelAIVibe,NovelAILineart,NovelAIPrompt,NovelAISketch,NovelAIDeclutter

# Map the node class to a unique name for registration
NODE_CLASS_MAPPINGS = {
    "NovelAI": NovelAINode,
    "NovelAI_VIBE": NovelAIVibe,
    "NovelAI_Lineart_Processor":NovelAILineart,
    "NovelAI_Sketch_Processor":NovelAISketch,
    "NovelAI_Declutter_Preprocessor":NovelAIDeclutter,
    "NovelAI_Prompt":NovelAIPrompt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NovelAI": "NovelAI Image Generate",
    "NovelAI_VIBE": "NovelAI Apply Vibe",
    "NovelAI_Lineart_Processor":"NovelAI Lineart Processor",
    "NovelAI_Sketch_Processor":"NovelAI Sketch Processor",
    "NovelAI_Declutter_Preprocessor":"NovelAI Declutter Preprocessor",
    "NovelAI_Prompt":"NovelAI Prompt Convert Tool"
}
print("----------------------Init Nai Nodes-----------------------")
__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS']