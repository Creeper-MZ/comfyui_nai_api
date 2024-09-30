from .nodes import NovelAINode,NovelAIVibe

# Map the node class to a unique name for registration
NODE_CLASS_MAPPINGS = {
    "NovelAI": NovelAINode,
    "NovelAI_VIBE": NovelAIVibe
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NovelAI": "NovelAI Image Generate",
    "NovelAI_VIBE": "NovelAI Apply Vibe"
}
print("Init Nai-----------------------")
__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS']