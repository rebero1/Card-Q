'''

The  following model is used to indetify between cat and  a dog in a picture

The datasets can be found on kaggle and can be replaced  in the placeholder location

We shall be using functional programminig to have  flexibility in desingning our model .


'''



from keras.preprocessing import image
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
import six
import keras
import os
from keras import layers
from keras import activations as acti
from keras import  initializers  as init
import numpy as np




def  model_making():
  input_shape = keras.Input(shape=(150,150,3))
  # layer1
  conv16_2 = layers.Conv2D(
      16, (2, 2), kernel_initializer="he_normal",padding="SAME")(input_shape)
  conv16_3 = layers.Conv2D(
      16, (3, 3), kernel_initializer="he_normal", padding="SAME")(input_shape)
  conv16_4 = layers.Conv2D(
      16, (4, 4), kernel_initializer="he_normal", padding="SAME")(input_shape)
  
  conv32_2 = layers.Conv2D(
      32, (2, 2), kernel_initializer="he_normal", padding="SAME")(input_shape)
  conv32_3 = layers.Conv2D(
      32, (3, 3), kernel_initializer="he_normal", padding="SAME")(input_shape)
  conv32_4 = layers.Conv2D(
      32, (4, 4), kernel_initializer="he_normal", padding="SAME")(input_shape)

  # adding layer1

  merge_16_2_32_4=layers.concatenate([conv16_2,conv32_4],axis=-1)
  merge_16_3_32_3 = layers.concatenate([conv16_3, conv32_3], axis=-1)
  merge_16_4_32_2 = layers.concatenate([conv16_4, conv32_2], axis=-1)
  merge_all = layers.concatenate([conv16_2,conv16_3,conv16_4,conv32_2,conv32_3,conv32_4],axis=-1)

  #appling maxpool 

  max_16_2_32_4=layers.MaxPooling2D(pool_size=(2,2))(merge_16_2_32_4)
  max_16_3_32_3 = layers.MaxPooling2D(pool_size=(2, 2))(merge_16_3_32_3)
  max_16_4_32_2 = layers.MaxPooling2D(pool_size=(2, 2))(merge_16_4_32_2)

  # applying bacth norm

  batch_1=layers.BatchNormalization()(max_16_2_32_4)
  batch_2 = layers.BatchNormalization()(max_16_3_32_3)
  batch_3 = layers.BatchNormalization()(max_16_4_32_2)

  # layer 2 
  conv64_2 = layers.Conv2D(64, (2, 2), kernel_initializer="he_normal", padding="SAME")(batch_1)
  conv64_3 = layers.Conv2D(64, (3, 3), kernel_initializer="he_normal" , padding="SAME")(batch_2)
  conv64_4 = layers.Conv2D(64, (4, 4), kernel_initializer="he_normal" , padding="SAME")(batch_3)
  # residual

  residual = layers.Conv2D(64,1,padding='same')(merge_all)
  residual=layers.MaxPooling2D(2)(residual)
  merge_resi_layer=layers.concatenate([residual,conv64_2,conv64_3,conv64_4],axis=-1)

  # max
  max_merged= layers.MaxPooling2D((2,2))(merge_resi_layer)
  bach_max = layers.BatchNormalization()(max_merged)

  # third layer
  conv128=layers.Conv2D(128,(3,3),kernel_initializer="he_normal")(bach_max)
  conv18 = layers.Conv2D(
      128*4, (3, 3), kernel_initializer="he_normal")(conv128)



  flatten=layers.Flatten()(conv18)
  dense_1=layers.Dense(32,activation="relu")(flatten)
  dense_2= layers.Dense(1,activation="sigmoid")(dense_1)
  model = keras.Model(input_shape,dense_2)
  return model


#%%
# editing photos for training 

for ids,photo in enumerate(os.listdir("/Users/reberoprince/Downloads/Photos")):
  os.rename("/Users/reberoprince/Downloads/Photos/{}".format(photo),
            "/Users/reberoprince/Downloads/Photos/{}.jpg".format(ids))
  
    
#%%
data_gen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
rotation_range=40,width_shift_range=.2,
height_shift_range=.2,shear_range=.2,zoom_range=.2,horizontal_flip=True)

 # Provide  the location of the cats and dog directory. 
  
train = data_gen.flow_from_directory(
    "LOCATION",
       target_size=(150, 150), batch_size=3, class_mode='binary'
)

callbacks=[keras.callbacks.TensorBoard(log_dir="my_log_dir"),keras.callbacks.EarlyStopping(
  monitor='val_loss',patience=2
)]
model=model_making()
model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
loss="binary_crossentropy",metrics=['acc'])
history=model.fit_generator(train,epochs=5,verbose=-1,callbacks=callbacks,steps_per_epoch=1)
model.save("prf_face.h5")
#%%


def plot_graph(history):
  loss = history.history['loss']
   
  epochs = range(1, len(loss) + 1)
  plot.figure()
  plot.plot(epochs, loss, 'bo', label='Training loss')
 
  plot.title('Training and validation loss')
  plot.legend()
  plot.show()
plot_graph(history)


#%%
from keras.preprocessing import image
imag = 'image.jpg'
img = image.load_img(imag, target_size=(150, 150))
img = image.img_to_array(img)
 
img = np.expand_dims(img, axis=0)
print(img.shape)
model.predict(img)



