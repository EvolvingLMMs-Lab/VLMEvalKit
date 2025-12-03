import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
import sys
# TODO: Verify this relative path is correct for the project structure
vggt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vggt'))
if vggt_path not in sys.path:
    sys.path.append(vggt_path)
from vggt.models.vggt import VGGT
import numpy as np

class VGGTSpatialConfig(PretrainedConfig):
    model_type = "vggt_spatial_model"

    def __init__(
        self,
        weights_path: str = "VGGT-1B", # Default relative name
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the spatial config dict if we are loading from Cut3rSpatialConfig
        if config_dict.get("model_type") == "vggt_spatial_model": # Or check if it's a base VGGT config?
            # Assuming a base VGGT config might not have 'spatial_config'. Adjust logic if needed.
            # If loading from a parent config that wraps this, you might need:
            # config_dict = config_dict["spatial_config"]
            pass # Keep config_dict as is if it's already the correct type

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class VGGTSpatialPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VGGTSpatialConfig
    base_model_prefix = "vggt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

def prepare_input(pixel_values):
    # pixel_values: (F, C, H, W)
    # resize to 378x378
    pixel_values = nn.functional.interpolate(pixel_values, size=(378, 378), mode='bilinear')

    # convert to 0-1
    pixel_values = pixel_values * 0.5 + 0.5
    pixel_values = pixel_values.unsqueeze(0) # 1 frame 3 h w
    return pixel_values

class VGGT_Encoder(nn.Module):
    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__()
        # Load model using the path from the config
        rank0_print(f"Loading VGGT from: {config.weights_path}")
        self.vggt = VGGT.from_pretrained(config.weights_path)
        self.vggt.eval()
        for param in self.vggt.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None # Add for API consistency if needed, though not used here
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # FIXME: can't process batch
        # import pdb; pdb.set_trace()
        views = prepare_input(pixel_values=pixel_values)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=views.dtype):
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(views)

        # just need the patch tokens of the last layer
        spatial_feat = aggregated_tokens_list[-1] # num_frame, 1, num_patch, token_feature
        spatial_feat = spatial_feat.to(pixel_values.dtype)
        spatial_feat = spatial_feat.squeeze(0) # num_frame, num_patch, token_feature
        camera_token = spatial_feat[:, 0:1, :]
        patch_tokens = spatial_feat[:, ps_idx:, :]

        # for debug(visualize point cloud)
        # pts3d_pred, pts3d_conf = self.vggt.point_head(
        #                 aggregated_tokens_list, images=views, patch_start_idx=ps_idx
        #             )

        return (camera_token, patch_tokens)


class VGGT_SpatialTransformer(nn.Module):
    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = VGGT_Encoder(config=config, **kwargs)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs

class VGGT_SpatialModel(VGGTSpatialPreTrainedModel):
    config_class = VGGTSpatialConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["VGGTSpatialEncoderLayer"]

    def __init__(self, config: VGGTSpatialConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = VGGT_SpatialTransformer(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: VGGTSpatialConfig, **kwargs):
        # Use the provided config object
        model = cls(config=config, **kwargs)
        # Potentially load state dict here if pretrained_model_name_or_path points to a checkpoint
        # containing the wrapper model's state, not just the base VGGT model.
        # For now, it assumes the config dictates the base model loading within VGGT_SpatialTransformer.
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None # Add for API consistency
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # Pass point_cloud_output_paths if the underlying model needs it, though VGGT_SpatialTransformer doesn't use it currently
            # point_cloud_output_paths=point_cloud_output_paths
        )

class VGGTSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        # Dynamically determine the weights path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assumes vggt repo is cloned sibling to the main project repo
        # ../../../ -> goes up from multimodal_spatial_encoder -> model -> llava -> project root
        vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        # Adjust the relative path 'vggt/VGGT-1B' as needed based on where VGGT weights are stored
        default_weights_name = "VGGT-1B" # Or whatever the default identifier is
        dynamic_weights_path = os.path.join(vlm_3r_root, 'vggt', default_weights_name) # Construct full path

        # Create config with the dynamic path
        self.config = VGGTSpatialConfig(
            weights_path=dynamic_weights_path,
            # Pass other relevant configs from spatial_tower_cfg if needed
            # Example: output_attentions=getattr(spatial_tower_cfg, 'output_attentions', False),
        )

        self.spatial_tower_name = spatial_tower # Keep track of the logical name/identifier

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower} using weights from {self.config.weights_path}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif hasattr(spatial_tower_cfg, "mm_tunable_parts") and "mm_spatial_tower" in spatial_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            # Store the config even if not loading immediately
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        # Load the VGGT_SpatialModel using the config which contains the weights_path
        # The spatial_tower_name might be used here if it points to a checkpoint containing the *entire* VGGTSpatialModel state,
        # rather than just the base VGGT weights. Adjust logic based on how checkpoints are saved/loaded.
        # Assuming spatial_tower_name is mainly an identifier and config.weights_path points to base VGGT weights.
        rank0_print(f"Instantiating VGGT_SpatialModel with config pointing to: {self.config.weights_path}")
        self.spatial_tower = VGGT_SpatialModel.from_pretrained(
            pretrained_model_name_or_path=self.spatial_tower_name, # Or potentially self.config.weights_path if appropriate
            config=self.config,
            device_map=device_map
        )

        self.spatial_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Pass point_cloud_output_paths=None if needed by the forward signature
                image_forward_out = self.spatial_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # Assuming image_forward_out is now a tuple (camera_token, patch_features)
                # We likely only need the patch features for downstream tasks
                patch_features = image_forward_out[1] # Get patch features
                # image_feature = image_forward_out.last_hidden_state.to(image.dtype) # Original line
                image_feature = patch_features.to(image.dtype)
                image_features.append(image_feature)
        else:
            # Pass point_cloud_output_paths=None if needed by the forward signature
            image_features = self.spatial_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
