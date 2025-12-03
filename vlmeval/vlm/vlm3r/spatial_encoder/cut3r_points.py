import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
from src.dust3r.model import ARCroco3DStereo
import numpy as np

class Cut3rPointsConfig(PretrainedConfig):
    model_type = "cut3r_points_model"

    def __init__(
        self,
        weights_path="../CUT3R/src/cut3r_512_dpt_4_64.pth",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the spatial config dict if we are loading from Cut3rSpatialConfig
        if config_dict.get("model_type") == "cut3r_points":
            config_dict = config_dict["spatial_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class Cut3rPointsPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Cut3rPointsConfig
    base_model_prefix = "cut3r"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

def prepare_input(pixel_values):
    pixel_values = nn.functional.interpolate(pixel_values, size=(432, 432), mode='bilinear')
    pixel_values = pixel_values.unsqueeze(1) ## FIXME: the second dimension is the number of frames in one batch
    views = []
    for i in range(len(pixel_values)):
        view = {
            "img": pixel_values[i],
            "ray_map": torch.full(
                (
                    pixel_values[i].shape[0],
                    6,
                    pixel_values[i].shape[-2],
                    pixel_values[i].shape[-1],
                ),
                torch.nan,
            ).to(pixel_values[i].device),
            "true_shape": torch.tensor(pixel_values[i].shape[-2:]).to(pixel_values[i].device),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4).unsqueeze(0).to(pixel_values[i].device),
            "img_mask": torch.tensor(True).unsqueeze(0).to(pixel_values[i].device),
            "ray_mask": torch.tensor(False).unsqueeze(0).to(pixel_values[i].device),
            "update": torch.tensor(True).unsqueeze(0).to(pixel_values[i].device),
            "reset": torch.tensor(False).unsqueeze(0).to(pixel_values[i].device),
        }
        views.append(view)
    return views

class Cut3rPointsEncoder(nn.Module):
    def __init__(self, config: Cut3rPointsConfig, **kwargs):
        super().__init__()
        self.cut3r = ARCroco3DStereo.from_pretrained(config.weights_path)
        self.cut3r.eval()
        self.config = config
        for param in self.cut3r.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        views = prepare_input(pixel_values=pixel_values)
        shape, feat_ls, pos = self.cut3r._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self.cut3r._init_state(feat[0], pos[0])
        mem = self.cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        out_points = []
        for i in range(len(views)):
            feat_i = feat[i].to(pixel_values.dtype)
            pos_i = pos[i]
            if self.cut3r.pose_head_flag:
                global_img_feat_i = self.cut3r._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.cut3r.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = self.cut3r._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.cut3r.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.cut3r.dec_depth + 1
            head_input = [
                dec[0],
                dec[self.cut3r.dec_depth * 2 // 4][:, 1:],
                dec[self.cut3r.dec_depth * 3 // 4][:, 1:],
                dec[self.cut3r.dec_depth],
            ]
            res = self.cut3r._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            out_points.append(res["pts3d_in_other_view"])
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].to(pixel_values.dtype)
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        # # for debug
        # pts3d = []
        # colors = []
        # for i in range(len(views)):
        #     pts3d.append(ress[i]["pts3d_in_other_view"])
        #     colors.append(ress[i]["rgb"])
        # pts3d = torch.cat(pts3d, dim=0) # b h w 3
        # colors = torch.cat(colors, dim=0) # b h w 3
        # pts3d = pts3d.view(-1, 3).float().cpu().numpy()
        # colors = colors * 0.5 + 0.5
        # colors = colors.view(-1, 3).float().cpu().numpy()
        # #保存为点云
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts3d)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud("pts3d.ply", pcd)

        out_points = torch.stack(out_points, dim=0) # [frame, b, H, W, 3]
        out_points = rearrange(out_points, 'frame b H W c -> (b frame) H W c')
        return out_points

class Cut3rPointsTransformer(nn.Module):
    def __init__(self, config: Cut3rPointsConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = Cut3rPointsEncoder(config=config, **kwargs)

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

class Cut3rPointsSpatialModel(Cut3rPointsPreTrainedModel):
    config_class = Cut3rPointsConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["Cut3rPointsEncoderLayer"]

    def __init__(self, config: Cut3rPointsConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = Cut3rPointsTransformer(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: Cut3rPointsConfig, **kwargs):
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

class Cut3rPointsSpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        dynamic_weights_path = os.path.join(vlm_3r_root, 'CUT3R', 'src', 'cut3r_512_dpt_4_64.pth')

        self.config = Cut3rPointsConfig(weights_path=dynamic_weights_path)

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

        self.spatial_tower = Cut3rPointsSpatialModel.from_pretrained(self.spatial_tower_name, config=self.config, device_map=device_map)

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
            image_features = image_forward_outs.to(images.dtype)
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
