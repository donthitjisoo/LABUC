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

EndoViT (https://huggingface.co/egeozsoy/EndoViT) is the one exception --
Apache 2.0, publicly downloadable, no login/access request needed.

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


class EndoViTExtractor(FoundationExtractor):
    """MAE-pretrained ViT-Base/16 on endoscopy images. Public, non-gated
    (Apache 2.0) -- no Hugging Face login required."""
    name = "endovit"
    hf_id = "egeozsoy/EndoViT"
    # EndoViT's own dataset statistics (not ImageNet mean/std), per the model card.
    MEAN = [0.3464, 0.2280, 0.2228]
    STD = [0.2520, 0.2128, 0.2093]

    def __init__(self, device):
        from functools import partial
        from pathlib import Path

        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from timm.models.vision_transformer import VisionTransformer
        from torchvision import transforms

        model_dir = snapshot_download(repo_id=self.hf_id, revision="main")
        weights_path = Path(model_dir) / "pytorch_model.bin"

        # num_classes=0 drops the classification head so forward() returns the
        # pooled 768-dim embedding directly; the checkpoint only has MAE-encoder
        # weights anyway, so strict=False just ignores any head/decoder mismatch.
        self.model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,
        )
        # weights_only=False: this checkpoint (from the official egeozsoy/EndoViT
        # repo) bundles an argparse.Namespace of training args alongside the
        # model weights, which PyTorch >=2.6's default safe unpickler rejects.
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)["model"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device).eval()
        self.device = device
        self.embed_dim = 768

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

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
    "endovit": EndoViTExtractor,
    "endodino": EndoDINOExtractor,
}
