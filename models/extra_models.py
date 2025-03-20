from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone
import torch

class HF_models:
    def __init__(self, encoder_name):
        self.encoder = AutoModelForDepthEstimation.from_pretrained(encoder_name)
        config = self.encoder.config
        self.encoder = self.encoder.backbone
        self.encoder.config = config    
        image_processor = AutoImageProcessor.from_pretrained(encoder_name)
        self.encoder.default_cfg = dict()
        self.encoder.default_cfg["mean"] = image_processor.image_mean
        self.encoder.default_cfg["std"] = image_processor.image_std
        #self.encoder.config.apply_layer_norm = False

        # needed to adjust embeddings for fine-tuning
        if isinstance(self.encoder.config.backbone_config.image_size, int):
            self.encoder.config.backbone_config.image_size = (self.encoder.config.backbone_config.image_size,
                                                              self.encoder.config.backbone_config.image_size)
        if isinstance(self.encoder.config.backbone_config.patch_size, int):
            self.encoder.config.backbone_config.patch_size = (self.encoder.config.backbone_config.patch_size,
                                                              self.encoder.config.backbone_config.patch_size) 
        grid_size = tuple(round(size / self.encoder.config.backbone_config.patch_size[0]) 
                          for size in self.encoder.config.backbone_config.image_size)
        self.encoder.config.grid_size = grid_size

        self.encoder.config.num_prefix_tokens = 0
        for prefix_token in ['cls_token']:
            if prefix_token in dir(self.encoder.embeddings):
                self.encoder.config.num_prefix_tokens += 1
        self.encoder.num_prefix_tokens = self.encoder.config.num_prefix_tokens      
    
    def forward_features(self, x, features = 'last_layer'):
        #print("     Forward features HF_models")
        if isinstance(self.encoder, Dinov2Backbone):
            x = self.encoder.embeddings(x)
            #print('     Got embeddings:', x.shape)
            x = self.encoder.encoder(x)
            #x = self.encoder(x)
            if features == 'last_layer':
                x = x.last_hidden_state
                #print('     Got encoder:', x.shape)
                x = self.encoder.layernorm(x)
                #print('     Returning last layer with layernorm:', x.shape)
                return x
            # for now, we only support 'last_layer' 
            elif features == 'standard':
                x = torch.stack(x)
                x = x.reshape(1, -1, 1024)
                return x
            else:
                raise ValueError(f"features {features} not recognized")