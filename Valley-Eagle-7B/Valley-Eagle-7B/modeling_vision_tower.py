import torch
import torch.nn as nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers import PretrainedConfig

siglip_config = PretrainedConfig.from_dict(
    {
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }
)

qwen2vl_vit_config = PretrainedConfig.from_dict(
    {
        "depth": 32,
        "embed_dim": 1280,
        "hidden_act": "quick_gelu",
        "hidden_size": 3584,
        "in_channels": 3,
        "in_chans": 3,
        "mlp_ratio": 4,
        "model_type": "qwen2_vl",
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "_attn_implementation": "flash_attention_2",
        "_attn_implementation_internal": "flash_attention_2"
    }
)

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if "siglip-so400m-patch14-384" in vision_tower:
        # Eagle
        if getattr(vision_tower_cfg, "eagle_vision_tower", None) is not None:
            if getattr(vision_tower_cfg, "_vit_attn_implementation", None) is not None:
                qwen2vl_vit_config._attn_implementation = vision_tower_cfg._vit_attn_implementation
                qwen2vl_vit_config._attn_implementation_internal = vision_tower_cfg._vit_attn_implementation
            
            qwen2vl_vision_tower = Qwen2VisionTransformerPretrainedModel._from_config(qwen2vl_vit_config)
            
            if getattr(vision_tower_cfg, "navit_merger_hidden_dim", None) is not None:
                del qwen2vl_vision_tower.merger
                qwen2vl_vision_tower.merger = CustomPatchMerger(
                    vision_tower_cfg.hidden_size, 
                    context_dim=1280, 
                    hidden_dim=getattr(vision_tower_cfg, "navit_merger_hidden_dim", None)
                ) # random initialize
            qwen2vl_vision_tower.requires_grad_(False)
            
            # If only use navit, delete siglip_vision_tower
            if getattr(vision_tower_cfg, "only_navit", False):
                siglip_vision_tower = None
            else:
                siglip_vision_tower = SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            
            return siglip_vision_tower, qwen2vl_vision_tower
        # Non-Eagle
        else:
            siglip_vision_tower = SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            return siglip_vision_tower
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")

class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, cache_dir="./cache_dir"):
        super().__init__()
        self.is_loaded = False
        self.image_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            from transformers import SiglipVisionModel
            self.cfg_only = siglip_config
            self.vision_tower = SiglipVisionModel._from_config(siglip_config)  # dummy-load

    def load_model(self):
        from transformers import SiglipVisionModel
        self.vision_tower = SiglipVisionModel._from_config(siglip_config)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        assert self.select_feature == "cls_patch"
        image_features = torch.cat([image_forward_outs[:, :1, :], image_forward_outs], dim=1)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_feature = self.feature_select(image_forward_out.last_hidden_state).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
            )
            image_features = self.feature_select(image_forward_outs.last_hidden_state).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CustomPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.input_dim = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        return x