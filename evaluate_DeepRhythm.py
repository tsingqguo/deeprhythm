import keras
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import *
import numpy as np
from keras import backend as K
import os
import plotly.offline as py
import plotly.graph_objects as go

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class X_plus_Layer(Layer):
    def __init__(self, **kwargs):
        super(X_plus_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', initializer='zeros', trainable=True)
        super(X_plus_Layer, self).build(input_shape)

    def call(self, inpt_x):
        x, A = inpt_x
        x_diag = x
        for i in range(25-1):
            x_diag = K.concatenate([x_diag, x], axis=2)
        x_diag_channals = x_diag
        x_diag_channals = K.expand_dims(x_diag, axis=3)
        x_diag = K.expand_dims(x_diag, axis=3)
        for i in range(3-1):
            x_diag_channals = K.concatenate([x_diag_channals, x_diag], axis=3)
        
        x_mask = x
        width = 25
        for w in range(width-1):
            x_mask = K.concatenate([x_mask, x], axis=2)
        x_mask_channals = x_mask
        x_mask_channals = K.expand_dims(x_mask, axis=3)
        x_mask = K.expand_dims(x_mask, axis=3)
        for i in range(3-1):
            x_mask_channals = K.concatenate([x_mask_channals, x_mask], axis=3)

        a_part = multiply([x_diag, A])
        a_part = self.alpha * a_part

        b_part = self.beta * x_mask_channals

        ans = Add()([a_part, b_part])
        return ans

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 300, 25, 3)

deeprhythm_path = "/DATASET/__saika_data/analysis/model/ablation_model/sa_pa_ta_ma_e2e_0.9775.h5"
model = load_model(deeprhythm_path, custom_objects={'multiply':multiply, 'Add':Add, 'X_plus_Layer':X_plus_Layer})

data_meso_dir = "/DATASET/__saika_data/dfdc/model_result/Meso/"
data_stmap_dir = "/DATASET/__saika_data/dfdc/stmap/"

print("DeepRhythm ... ")

meso_data = []
stmap_data = []
y_data = []
for methname in ["method_A", "method_B", "original_videos"]:

    if methname in ["method_A", "method_B"]:
        y_lbl = 1
    else:
        y_lbl = 0

    mesomethpath = data_meso_dir + methname + '/'
    stmethpath = data_stmap_dir + methname + '/'

    vidlist = os.listdir(mesomethpath)
    vidlist.sort()
    for vidname in vidlist:
        meso = np.load(mesomethpath + vidname)
        meso = np.resize(meso, (300, 1))
        stmap = np.load(stmethpath + vidname[:-4] + ".avi.npy")

        meso_data.append(meso)
        stmap_data.append(stmap)
        y_data.append(y_lbl)

meso_data = np.array(meso_data)
stmap_data = np.array(stmap_data)
y_data = np.array(y_data)

score, acc = model.evaluate([stmap_data, meso_data], y_data,
                    batch_size=32)
print("DeepRhythm: {}".format(acc))

meso_data = []
stmap_data = []
y_data = []