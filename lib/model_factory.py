import torch

from lib.models.dense import Dense, DenseConfig
from lib.models.conv_small import ConvSmall, ConvSmallConfig
from lib.models.conv import Conv, ConvConfig
from lib.models.conv_lap import ConvLAP, ConvLAPConfig
from lib.models.transformer import Transformer, TransformerConfig
from lib.models.mlp import MLPClass, MLPClassConfig
from lib.models.mlp import MLP, MLPConfig
from lib.models.mlp_proj import MLPProjClass, MLPProjClassConfig
from lib.models.resnet import Resnet, ResnetConfig
from lib.models.swin_transformer_v2 import SwinTiny, SwinTinyConfig

from lib.models.llama2_generative import LLaMA2Generative, LLaMA2GenerativeConfig
from lib.models.mistral_generative import MistralGenerative, MistralGenerativeConfig

class _ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[DenseConfig.__name__] = Dense
        self.models[ConvSmallConfig.__name__] = ConvSmall
        self.models[ConvConfig.__name__] = Conv
        self.models[ConvLAPConfig.__name__] = ConvLAP
        self.models[TransformerConfig.__name__] = Transformer
        self.models[MLPConfig.__name__] = MLP
        self.models[MLPClassConfig.__name__] = MLPClass
        self.models[MLPProjClassConfig.__name__] = MLPProjClass
        self.models[ResnetConfig.__name__] = Resnet
        self.models[SwinTinyConfig.__name__] = SwinTiny
        self.models[LLaMA2GenerativeConfig.__name__] = LLaMA2Generative
        self.models[MistralGenerativeConfig.__name__] = MistralGenerative

    def register(self, config_class, model_class):
        self.models[config_class.__name__] = model_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__.__name__](model_config, data_config)

    def get_class(self, model_config):
        return self.models[model_config.__class__.__name__]


_model_factory = None


def get_factory():
    global _model_factory
    if _model_factory is None:
        _model_factory = _ModelFactory()

    return _model_factory
