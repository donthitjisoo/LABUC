"""Pretrained foundation-model feature extractors for LIMUC embedding visualization.

These are large, third-party, GATED models on Hugging Face. Before using any
of them you must, under your own HF account:
  1. Visit the model page and accept its license / access request.
  2. Authenticate locally: `huggingface-cli login` (or set the HF_TOKEN env var).
Model pages:
  - DINOv3 ViT-L/16: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
  - MedSigLIP-448:   https://huggingface.co/google/medsiglip-448
  - UNI2-h:          https://huggingface.co/MahmoodLab/UNI2-h  (approval requires
                      your HF account email to match an institutional address)

EndoDINO (arXiv:2501.05488) has NO public checkpoint release as of this
writing -- EndoDINOExtractor is a stub; fill it in if you obtain private
weights directly from the authors.
"""
import torch
from PIL import Image


class FoundationExtractor:
    """Common interface: .preprocess(pil_image) -> tensor, .embed(batch) -> (B, D) tensor."""
    name: str
    embed_dim: int

    def preprocess(self, image: Image.Image):
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DINOv3Extractor(FoundationExtractor):
    name = "dinov3_vitl16"
    hf_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id).to(device).eval()
        self.device = device
        self.embed_dim = self.model.config.hidden_size

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(self, batch):
        out = self.model(pixel_values=batch.to(self.device))
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]


class MedSigLIPExtractor(FoundationExtractor):
    name = "medsiglip_448"
    hf_id = "google/medsiglip-448"

    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id).to(device).eval()
        self.device = device
        self.embed_dim = self.model.config.vision_config.hidden_size

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(self, batch):
        out = self.model.get_image_features(pixel_values=batch.to(self.device))
        if isinstance(out, torch.Tensor):
            return out
        # Some transformers versions return the full vision-model output here
        # instead of a bare pooled tensor -- unwrap it defensively.
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]


class UNI2Extractor(FoundationExtractor):
    name = "uni2_h"
    hf_id = "MahmoodLab/UNI2-h"

    def __init__(self, device):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        # Required per the UNI2-h model card -- without these, timm's hub auto-config
        # picks the wrong base architecture (e.g. vit_giant_patch14_224) and the
        # checkpoint's position embeddings fail to resample correctly.
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = timm.create_model(f"hf-hub:{self.hf_id}", pretrained=True, **timm_kwargs).to(device).eval()
        config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = create_transform(**config)
        self.device = device
        self.embed_dim = 1536

    def preprocess(self, image):
        return self.transform(image)

    @torch.no_grad()
    def embed(self, batch):
        return self.model(batch.to(self.device))


class EndoDINOExtractor(FoundationExtractor):
    """Stub: EndoDINO has no public checkpoint. Point --endodino-checkpoint at
    weights you've obtained directly from the paper's authors and adapt
    __init__/preprocess/embed below to match their actual release format
    (architecture, input size, and normalization stats aren't public yet)."""
    name = "endodino"

    def __init__(self, device, checkpoint_path=None):
        if not checkpoint_path:
            raise RuntimeError(
                "EndoDINO has no public checkpoint release (see arXiv:2501.05488). "
                "Pass --endodino-checkpoint pointing at weights obtained from the "
                "authors, and fill in EndoDINOExtractor to match their format."
            )
        raise NotImplementedError(
            "Loading logic isn't implemented -- EndoDINO's released weight format "
            "isn't public. Adapt this method once you have the actual checkpoint."
        )


REGISTRY = {
    "dinov3_vitl16": DINOv3Extractor,
    "medsiglip_448": MedSigLIPExtractor,
    "uni2_h": UNI2Extractor,
    "endodino": EndoDINOExtractor,
}
