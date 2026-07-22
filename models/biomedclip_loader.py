"""Load BiomedCLIP and optionally install the project-local VPT adapter."""

import inspect

import timm
import transformers
from timm.models.vision_transformer import VisionTransformer

from open_clip.src.open_clip import create_model_from_pretrained
from open_clip.src.open_clip.model import CustomTextCLIP
from open_clip.src.open_clip.timm_model import TimmModel

from .vpt import TimmViTVisualPromptEncoder


BIOMEDCLIP_MODEL_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


def _source_path(obj):
    try:
        return inspect.getfile(obj)
    except (OSError, TypeError):
        return "unknown"


def _validate_biomedclip(model):
    if not isinstance(model, CustomTextCLIP):
        raise TypeError(
            "Expected OpenCLIP CustomTextCLIP, got {}".format(type(model).__name__)
        )
    if not isinstance(model.visual, TimmModel):
        raise TypeError(
            "VPT requires OpenCLIP TimmModel, got {}".format(
                type(model.visual).__name__
            )
        )
    if not isinstance(model.visual.trunk, VisionTransformer):
        raise TypeError(
            "VPT requires timm VisionTransformer, got {}".format(
                type(model.visual.trunk).__name__
            )
        )


def load_biomedclip(vpt_enabled=False, vpt_mode="shallow", vpt_num_tokens=5, vpt_dropout=0.0):
    """Load the original weights and optionally wrap only the visual forward."""
    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL_ID)

    print("BiomedCLIP model source: {}".format(_source_path(CustomTextCLIP)))
    print("OpenCLIP timm adapter source: {}".format(_source_path(TimmModel)))
    print("timm VisionTransformer source: {}".format(_source_path(VisionTransformer)))
    print("timm version: {}".format(timm.__version__))
    print("transformers version: {}".format(transformers.__version__))

    if vpt_enabled:
        _validate_biomedclip(model)
        model.visual = TimmViTVisualPromptEncoder(
            model.visual,
            num_tokens=vpt_num_tokens,
            mode=vpt_mode,
            dropout=vpt_dropout,
        )

    return model, preprocess
