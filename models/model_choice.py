from models.resnet import ResNet18
from models.resnet_de_channal import ResNet_de_channel
from models.resnet_stage_ratio import (
    ResNet_stage_ratio1131,
    ResNet_stage_ratio2242,
    ResNet_stage_ratio2262,
    ResNet_stage_ratio3333,
)
from models.resnet_de_channal_patchify import ResNet_de_channel_patchify
from models.resnet_de_channal_conv2by2_down import ResNet_de_channel_conv2by2_down
from models.resnet_depth_conv import ResNet_de_channel_depth_conv
from models.resnet_de_resblock import ResNet_de_resblock
from models.resnet_de_resblock_stage_ratio import (
    ResNet_de_resblock321,
    ResNet_de_resblock141,
)

model_list = {
    "resnet18": ResNet18(),
    "resnet_de_channel": ResNet_de_channel(),
    "resnet_stage_ratio_1131": ResNet_stage_ratio1131(),
    "resnet_stage_ratio_2242": ResNet_stage_ratio2242(),
    "resnet_stage_ratio_2262": ResNet_stage_ratio2262(),
    "resnet_stage_ratio_3333": ResNet_stage_ratio3333(),
    "resnet_de_channal_patchify": ResNet_de_channel_patchify(),
    "resnet_de_channal_conv2by2_down": ResNet_de_channel_conv2by2_down(),
    "resnet_de_channal_depth_conv": ResNet_de_channel_depth_conv(),
    "resnet_de_resblock": ResNet_de_resblock(),
    "resnet_de_resblock321": ResNet_de_resblock321(),
    "resnet_de_resblock141": ResNet_de_resblock141(),
}


def model_choice(model_name):
    return model_list[model_name]
