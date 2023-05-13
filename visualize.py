from collections import defaultdict

import visualkeras
from PIL import ImageFont
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D, ReLU)

from processing.grey_world.cc_layers import GreyWorld
from src.model import get_model

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = '#E69F00'
color_map[ReLU]['fill'] = '#D55E00'
color_map[MaxPooling2D]['fill'] = '#56B4E9'
color_map[Flatten]['fill'] = '#CC79A7'
color_map[Dropout]['fill'] = 'black'
color_map[Dense]['fill'] = '#F0E442'
color_map[Sequential]['fill'] = '#009E73'
color_map[BatchNormalization]['fill'] = '#0072B2'

# using comic sans is strictly prohibited!
font = ImageFont.truetype("arial.ttf", 45)

model = get_model(num_classes=17, use_batchnorm=True)

model.summary()

visualkeras.layered_view(model, to_file='out/model.png', spacing=40,
                         legend=True, font=font, color_map=color_map).show()
