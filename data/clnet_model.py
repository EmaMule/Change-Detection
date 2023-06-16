from keras.layers import Add,ZeroPadding2D,Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
from keras.models import Model

def Left_Block(input_tensor,outplane,mode="None"):
    x = Basic_block(input_tensor, outplane,mode=mode)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    return x

def Right_Block(input_tensor,outplane,mode="None"):
    x=Basic_block_right(input_tensor,outplane,mode=mode)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    return x

def Basic_block(input_tensor,output_tensor,mode="None"):
    x=Conv2D(output_tensor,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu"
             )(input_tensor)
    x0=x
    x=BatchNormalization()(x)
    #x = GroupNormalization()(x)
    x = Conv2D(output_tensor, kernel_size=(3, 3), strides=(1, 1), padding="same",activation="relu"
               )(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization()(x)
    if mode=="None":
        return x
    if mode =="residual":
        x=Add()([x,x0])
    return x
def Basic_block_right(input_tensor,output_tensor,mode="None"):
    x=ZeroPadding2D(padding=(1,1),data_format=None)(input_tensor)
    x=Conv2D(output_tensor,kernel_size=(3,3),strides=(2,2),padding="same",activation="relu"
             )(x)
    x0=x
    x=BatchNormalization()(x)
    #x = GroupNormalization()(x)
    x = Conv2D(output_tensor, kernel_size=(3, 3), strides=(1, 1), padding="same",activation="relu"
               )(x)
    x=BatchNormalization()(x)
    #x = GroupNormalization()(x)
    if mode=="None":
        return x
    if mode =="residual":
        x=Add()([x,x0])
    return x


def CLN(input_shape,num_class=1):
    nb_filter = 24
    a = nb_filter
    inputs = Input(shape=input_shape)
    con_axis = 3
    conv1_1 = Left_Block(inputs, a, mode="None")
    conv2_1 = Left_Block(conv1_1, 2 * a)
    conv2_2 = Right_Block(inputs, a, mode="None")
    conv2_3 = concatenate([conv2_1, conv2_2], axis=con_axis)
    conv3_1 = Right_Block(conv1_1, 2 * a)
    conv3_2 = Left_Block(conv2_3, 6 * a)
    conv3_3 = concatenate([conv3_1, conv3_2], axis=con_axis)
    conv4_1 = Left_Block(conv3_2,12*a)
    conv4_2 = Right_Block(conv2_3, 6 * a)
    conv4_3 = concatenate([conv4_1, conv4_2], axis=con_axis)
    conv4_3=Conv2D(filters=6*a,kernel_size=1,padding="same",strides=1)(conv4_3)
    conv4_4=Left_Block(conv3_3,16*a)
    conv4_4=concatenate([conv4_3,conv4_4],axis=con_axis)
    conv_bottom = Basic_block(conv4_4, 2 * 8 * a)
    up423 = Conv2DTranspose(8 * a, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv_bottom)
    #up423 = BatchNormalization()(up423)
    up3 = concatenate([conv3_3, up423], axis=con_axis)
    #up3 = Activation("relu")(up3)
    conv3_4 = Basic_block(up3, 2 * 3 * a)
    up322 = Conv2DTranspose(3 * a, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv3_4)
    #up322 = BatchNormalization()(up322)
    up2 = concatenate([conv2_3, up322], axis=con_axis)
    #up2 = Activation("relu")(up2)
    conv2_4 = Basic_block(up2, 2 * a)
    up221 = Conv2DTranspose(a, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv2_4)
    #up221 = BatchNormalization()(up221)
    up1 = concatenate([conv1_1, up221], axis=con_axis)
    #up1 = Activation("relu")(up1)
    conv1_2 = Basic_block(up1, a)
    up0 = Conv2DTranspose(a, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv1_2)
    #up0 = BatchNormalization()(up0)
    strangenet_output = Conv2D(num_class, kernel_size=(1, 1), activation="sigmoid", padding="same", name='output')(up0)
    model = Model(inputs, [strangenet_output])
    # model.compile(optimizer=Adam(lr=1e-4), loss=weighted_bce_dice_loss,
    #               metrics=['accuracy'])
    model.summary()
    return model
