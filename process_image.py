import tensorflow as tf
import tensorflow_hub as hub

def process_image(npimage):
    tfimage = tf.convert_to_tensor(npimage)
    image_size = 224
    tfimageresized = tf.image.resize(tfimage, (image_size, image_size))
    tfimageresized /= 255
    npimageout =tfimageresized.numpy()
    #print(npimageout.shape)
    return npimageout
