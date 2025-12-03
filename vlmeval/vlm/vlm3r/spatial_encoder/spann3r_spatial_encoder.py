import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
from spann3r.model import Spann3R, SpatialMemory

class Spann3rSpatialConfig(PretrainedConfig):
    model_type = "spann3r_spatial_model"

    def __init__(
        self,
        dust3r_name="../spann3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        weights_path="../spann3r/checkpoints/spann3r.pth",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dust3r_name = dust3r_name
        self.weights_path = weights_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the spatial config dict if we are loading from Spann3rSpatialConfig
        if config_dict.get("model_type") == "spann3r":
            config_dict = config_dict["spatial_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class Spann3rSpatialPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Spann3rSpatialConfig
    base_model_prefix = "spann3r"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

class Spann3REncoder(nn.Module):
    def __init__(self, config: Spann3rSpatialConfig, **kwargs):
        super().__init__()
        self.spann3r = Spann3R(dus3r_name=config.dust3r_name, use_feat=False)
        self.spann3r.load_state_dict(torch.load(config.weights_path)['model'])
        self.spann3r.eval()
        for param in self.spann3r.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        frames = [{'img': pixel_values[i]} for i in range(len(pixel_values))]
        if self.training:
            sp_mem = SpatialMemory(self.spann3r.norm_q, self.spann3r.norm_k, self.spann3r.norm_v, mem_dropout=self.spann3r.mem_dropout, attn_thresh=0)
        else:
            sp_mem = SpatialMemory(self.spann3r.norm_q, self.spann3r.norm_k, self.spann3r.norm_v)

        feat1, feat2, pos1, pos2, shape1, shape2 = None, None, None, None, None, None
        feat_k1, feat_k2 = None, None

        preds = None
        preds_all = []
        spatial_feat = []
        for i in range(len(frames)):
            if i == len(frames)-1:
                break
            view1 = frames[i]
            view2 = frames[(i+1)]

            ##### Encode frames
            # feat1: [bs, p=196, c=1024]
            feat1, feat2, pos1, pos2, shape1, shape2 = self.spann3r.encode_frames(view1, view2, feat1, feat2, pos1, pos2, shape1, shape2)

            ##### Memory readout
            if feat_k2 is not None:
                feat_fuse = sp_mem.memory_read(feat_k2, res=True)
                # feat_fuse = feat_fuse + feat1
            else:
                feat_fuse = feat1

            ##### Decode features
            # dec1[-1]: [bs, p, c=768]
            dec1, dec2 = self.spann3r.decode(feat_fuse, pos1, feat2, pos2)

            ##### Encode feat key
            feat_k1 = self.spann3r.encode_feat_key(feat1, dec1[-1], 1)
            feat_k2 = self.spann3r.encode_feat_key(feat2, dec2[-1], 2)

            ##### Regress pointmaps
            with torch.cuda.amp.autocast(enabled=True):
                res1 = self.spann3r.dust3r._downstream_head(1, [tok for tok in dec1], shape1)
                res2 = self.spann3r.dust3r._downstream_head(2, [tok for tok in dec2], shape2)
            res1['pts3d'] = res1['pts3d'].to(device=pixel_values.device, dtype=pixel_values.dtype)
            res2['pts3d'] = res2['pts3d'].to(device=pixel_values.device, dtype=pixel_values.dtype)
            ##### Memory update
            cur_v = self.spann3r.encode_cur_value(res1, dec1, pos1, shape1)

            if self.training:
                sp_mem.add_mem(feat_k1, cur_v+feat_k1)
            else:
                sp_mem.add_mem_check(feat_k1, cur_v+feat_k1)

            res2['pts3d_in_other_view'] = res2.pop('pts3d')

            if preds is None:
                preds = [res1]
                preds_all = [(res1, res2)]
            else:
                res1['pts3d_in_other_view'] = res1.pop('pts3d')
                preds.append(res1)
                preds_all.append((res1, res2))

            spatial_feat.append(dec1[-1])

        preds.append(res2)
        spatial_feat.append(dec2[-1])
        spatial_feat = torch.stack(spatial_feat, dim=0) # [n, b, token_num, token_dim]
        spatial_feat = rearrange(spatial_feat, 'n b token_num token_dim -> (b n) token_num token_dim')

        # just return the spatial feature(last hidden state)
        return (spatial_feat, None, None, None)


class Spann3rSpatialTransformer(nn.Module):
    def __init__(self, config: Spann3rSpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = Spann3REncoder(config=config, **kwargs)

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

        last_hidden_state = encoder_outputs[0]

        if not return_dict:
            return (last_hidden_state, None) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )

class Spann3rSpatialModel(Spann3rSpatialPreTrainedModel):
    config_class = Spann3rSpatialConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Spann3rSpatialEncoderLayer"]

    def __init__(self, config: Spann3rSpatialConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = Spann3rSpatialTransformer(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config = Spann3rSpatialConfig()
        model = cls(config=config, **kwargs)
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class Spann3rSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        self.config = Spann3rSpatialConfig()

        self.spatial_tower_name = spatial_tower

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif hasattr(spatial_tower_cfg, "mm_tunable_parts") and "mm_spatial_tower" in spatial_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        self.spatial_tower = Spann3rSpatialModel.from_pretrained(self.spatial_tower_name, device_map=device_map)

        self.spatial_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.spatial_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.last_hidden_state.to(image.dtype)
                # assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.spatial_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_outs.last_hidden_state.to(images.dtype)
            # assert image_features.shape[-2] == 729

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
