import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization
import numpy as np
from PIL import Image, ImageFilter

#Load in Models
class MiniImagenetModel(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MiniImagenetModel'

        super(MiniImagenetModel, self).__init__(*args, **kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        # self.bn1 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        # self.bn2 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        # self.bn3 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        # self.bn4 = tf.keras.layers.LayerNormalization(center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')

        self.dense = Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def get_features(self, inputs, training=False):
        import numpy as np
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        f = self.flatten(c4)
        return f

    def call(self, inputs, training=False):
        f = self.get_features(inputs, training=training)
        out = self.dense(f)

        return out


# Omniglot
class SimpleModel(tf.keras.Model):
    name = 'SimpleModel'

    def __init__(self, num_classes):
        super(SimpleModel, self).__init__(name='simple_model')

        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1', strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2', strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3', strides=(2, 2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4',  strides=(2, 2), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization(center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reduce_mean(c4, [1, 2])
        f = self.flatten(c4)
        out = self.dense(f)

        return out

# latest = './miniImageNetModel/model.ckpt-60000'
latest = './omniglotModel/model.ckpt-2000'

# model = MiniImagenetModel(num_classes=5)

model = SimpleModel(num_classes=5)


model.load_weights(latest)
model.compile()

numPics = 0

#Modify images for model input (For Omniglot)
imageSource = [Image.open('./image0.png').convert('L'), Image.open('./image1.png').convert('L'),
               Image.open('./image2.png').convert('L'), Image.open('./image3.png').convert('L'),
               Image.open('./image4.png').convert('L')]

images = [np.array(imageSource[0].getdata()).reshape(imageSource[0].size[0], imageSource[0].size[1], 1),
          np.array(imageSource[1].getdata()).reshape(imageSource[1].size[0], imageSource[1].size[1], 1),
          np.array(imageSource[2].getdata()).reshape(imageSource[2].size[0], imageSource[2].size[1], 1),
          np.array(imageSource[3].getdata()).reshape(imageSource[3].size[0], imageSource[3].size[1], 1),
          np.array(imageSource[4].getdata()).reshape(imageSource[4].size[0], imageSource[4].size[1], 1)]

imageInput = np.stack(images, axis=0)

# Fine Tune Model to picture inputs
n=5
k=1
labels = np.repeat(np.arange(n),k) # n=5, k=1
labels = np.float32(labels)
imageInput = np.float32(imageInput)
model.fit(imageInput, labels, epochs=10)

# Make Predictions
testImageSource = Image.open('./imageTest.png').convert('L')
testImage = np.array(testImageSource.getdata()).reshape(testImageSource.size[1], testImageSource.size[1], 1)
testImage = np.float32(testImage)
prediction = model.predict([testImage])
print(prediction)
