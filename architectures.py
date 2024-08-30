# Various architectures that can be used as embedding model for the siamese network

import tensorflow as tf
import keras
from tensorflow.keras import backend as K


class DistanceLayer(tf.keras.layers.Layer):

    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, anchor, positive, negative):
        ap_distance = tf.sqrt(K.sum(tf.square(anchor - positive), -1))
        an_distance = tf.sqrt(K.sum(tf.square(anchor - negative), -1))

        return (ap_distance, an_distance)
    
class L2_layer(tf.keras.layers.Layer):

    def __init__(self, axis=-1, epsilon=1e-12, **kwargs):
        super(L2_layer, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        super(L2_layer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        norm = tf.norm(inputs, axis=self.axis, keepdims=True)
        return tf.math.divide_no_nan(inputs, norm + self.epsilon)
  

def Vanilla(input_shape, n_feature_maps, activation, kernels_size,optimizer, droputRate, output_size):

    '''
    vanilla CNN
    INPUT : Only Signal
    
    '''

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    output_block = keras.layers.Activation(activation)(conv_z)

    if droputRate > 0:
        output_block = keras.layers.Dropout(droputRate)(output_block)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block)
    gap_layer = keras.layers.Flatten()(gap_layer)
    output_layer = keras.layers.Dense(256, activation=activation)(gap_layer)
    output_layer = keras.layers.Dense(output_size, activation='linear')(gap_layer)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=input_layer, outputs=output_layer) 

    ##build siamese model
    anchor   =  keras.layers.Input(input_shape)
    positive =  keras.layers.Input(input_shape)
    negative =  keras.layers.Input(input_shape)

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese


def Resnet_GA_1(input_shape, input_shapeGA, n_feature_maps,activation, kernels_size,optimizer, droputRate, output_size):

    '''
    Single Block Resnet
    INPUT : Signal and Gestational Age (single number)
    '''
      
    input_layer = keras.layers.Input(input_shape)
    inputGA = keras.layers.Input(input_shapeGA)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation(activation)(output_block_1)
    
    if droputRate > 0:
        output_block_1 = keras.layers.Dropout(droputRate)(output_block_1)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_1) #era output_block_3

    #added to pass the GA
    fcl_concat = keras.layers.Concatenate()([gap_layer, inputGA])
    FCL = keras.layers.Dense(42, activation=activation)(fcl_concat)
    if droputRate > 0:
        FCL = keras.layers.Dropout(droputRate)(FCL)
    FCL = keras.layers.Flatten()(FCL)
    output_layer = keras.layers.Dense(output_size, activation='linear')(FCL)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=[input_layer,inputGA], outputs=output_layer)

    #build siamese model
    anchor = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    positive = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    negative = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese

def Resnet_GA_2(input_shape, input_shapeGA, n_feature_maps,activation, kernels_size,optimizer, droputRate, output_size):

    '''
    Two Blocks Resnet
    INPUT : Signal and Gestational Age (single number)
    '''
      
    input_layer = keras.layers.Input(input_shape)
    inputGA = keras.layers.Input(input_shapeGA)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation(activation)(output_block_1)
    if droputRate > 0:
        output_block_1 = keras.layers.Dropout(droputRate)(output_block_1)

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation(activation)(output_block_2)
    if droputRate > 0:
        output_block_2 = keras.layers.Dropout(droputRate)(output_block_2)


    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_2) #era output_block_3

    #added to pass the GA
    fcl_concat = keras.layers.Concatenate()([gap_layer, inputGA])
    FCL = keras.layers.Dense(42, activation=activation)(fcl_concat)
    if droputRate > 0:
        FCL = keras.layers.Dropout(droputRate)(FCL)
    FCL = keras.layers.Flatten()(FCL)
    output_layer = keras.layers.Dense(output_size, activation='linear')(FCL)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=[input_layer,inputGA], outputs=output_layer)

    #build siamese model
    anchor = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    positive = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    negative = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese


def Resnet_GA_4(input_shape, input_shapeGA, n_feature_maps,activation, kernels_size,optimizer, droputRate, output_size):

    '''
    Four Blocks Resnet
    INPUT : Signal and Gestational Age (single number)
    #TODO: check with Federica that the last block is correct
    '''

    input_layer = keras.layers.Input(input_shape)
    inputGA = keras.layers.Input(input_shapeGA)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation(activation)(output_block_1)
    

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation(activation)(output_block_2)

    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[0], padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=1, padding='same')(output_block_2)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation(activation)(output_block_3)
     
    # BLOCK 4
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[0], padding='same')(output_block_3)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 3, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z) 
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)
    
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_3)
    output_block_4 = keras.layers.add([shortcut_y, conv_z])

    output_block_4 = keras.layers.Activation(activation)(output_block_4)
    if droputRate > 0:
        output_block_4 = keras.layers.Dropout(droputRate)(output_block_4)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_4)

    #added to pass the GA
    fcl_concat = keras.layers.Concatenate()([gap_layer, inputGA])
    FCL = keras.layers.Dense(42, activation=activation)(fcl_concat)
    if droputRate > 0:
        FCL = keras.layers.Dropout(droputRate)(FCL)
    output_layer = keras.layers.Dense(output_size, activation='linear')(FCL)
    
    #L2 normalization
    output_layer = L2_layer()(output_layer)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=[input_layer,inputGA], outputs=output_layer)

    Embeddingmodel.summary()    

    #build siamese model
    anchor = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    positive = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    negative = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese


def Resnet_GA_3(input_shape, input_shapeGA, n_feature_maps,activation, kernels_size,optimizer, droputRate, output_size):

    '''
    Three Blocks Resnet
    INPUT : Signal and Gestational Age (single number)
    '''
      
    input_layer = keras.layers.Input(input_shape)
    inputGA = keras.layers.Input(input_shapeGA)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation(activation)(output_block_1)
    

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation(activation)(output_block_2)
     
    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z) 
    if droputRate > 0:
        conv_z = keras.layers.Dropout(droputRate)(conv_z)
    
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])

    output_block_3 = keras.layers.Activation(activation)(output_block_3)
    if droputRate > 0:
        output_block_3 = keras.layers.Dropout(droputRate)(output_block_3)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    #added to pass the GA
    fcl_concat = keras.layers.Concatenate()([gap_layer, inputGA])
    FCL = keras.layers.Dense(42, activation=activation)(fcl_concat)
    if droputRate > 0:
        FCL = keras.layers.Dropout(droputRate)(FCL)
    output_layer = keras.layers.Dense(output_size, activation='linear')(FCL)
    
    #L2 normalization
    output_layer = L2_layer()(output_layer)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=[input_layer,inputGA], outputs=output_layer, name = 'EmbeddingModel')

    Embeddingmodel.summary()    

    #build siamese model
    anchor = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    positive = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]
    negative = [keras.layers.Input(input_shape), keras.layers.Input(input_shapeGA)]

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese

  

def Resnet_3(input_shape, n_feature_maps, activation, kernels_size,optimizer, droputRate, output_size):

    '''
    Three Blocks Resnet
    INPUT : Only Signals 
    '''

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[0], padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    if droputRate > 0:
        conv_y = keras.layers.Dropout(droputRate)(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation(activation)(output_block_1)

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    if droputRate > 0:
        conv_x = keras.layers.Dropout(droputRate)(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    if droputRate > 0:
        conv_y = keras.layers.Dropout(droputRate)(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation(activation)(output_block_2)

    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[0], padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    if droputRate > 0:
        conv_x = keras.layers.Dropout(droputRate)(conv_x)
    conv_x = keras.layers.Activation(activation)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[1], padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    if droputRate > 0:
        conv_y = keras.layers.Dropout(droputRate)(conv_y)
    conv_y = keras.layers.Activation(activation)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=kernels_size[2], padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    if droputRate > 0:
        output_block_3 = keras.layers.Dropout(droputRate)(output_block_3)
    output_block_3 = keras.layers.Activation(activation)(output_block_3)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    
    gap_layer = keras.layers.Flatten()(gap_layer)
    output_layer = keras.layers.Dense(output_size, activation='linear')(gap_layer)

    #build embedding model
    Embeddingmodel = keras.models.Model(inputs=input_layer, outputs=output_layer, name = 'EmbeddingModel') 

    ##build siamese model
    anchor   =  keras.layers.Input(input_shape)
    positive =  keras.layers.Input(input_shape)
    negative =  keras.layers.Input(input_shape)

    outputA = Embeddingmodel(anchor)
    outputP = Embeddingmodel(positive)
    outputN = Embeddingmodel(negative)

    embedded_distance = DistanceLayer()(outputA, outputP, outputN)

    siamese = keras.Model(inputs=[anchor,positive,negative], outputs=embedded_distance)

    siamese.compile(optimizer=optimizer)
    
    return siamese







