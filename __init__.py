"""
ComfyUI Custom Node: Perspective to Equirectangular with Blending

Provides nodes for projecting perspective images onto equirectangular panoramas
with seamless blending.

Installation:
    Place this folder in ComfyUI/custom_nodes/
    Restart ComfyUI

Author: Claude
License: MIT
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
