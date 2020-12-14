import tensorflow as tf


def load_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.bfloat16)
    img = img[tf.newaxis, :]
    return img

def resize_img(img, min_size=512): # resize + keep aspect ratio
    width, height, _ = tf.unstack(tf.shape(img), num=3)
    if height < width:
        new_height = min_size
        new_width = int(width * new_height / height)
    else:
        new_width = min_size
        new_height = int(height * new_width / width)
    img = tf.image.resize(img, size=(new_width, new_height))
    return img


def mean_std_loss(feat, feat_stylized, epsilon=1e-5):
    feat_mean, feat_var = tf.nn.moments(feat, axes=[1,2])
    feat_stylized_mean, feat_stylized_var = tf.nn.moments(feat_stylized, axes=[1,2])

    feat_std = tf.math.sqrt(feat_var + epsilon)
    feat_stylized_std = tf.math.sqrt(feat_stylized_var + epsilon)

    loss = tf.losses.mse(feat_stylized_mean, feat_mean) + tf.losses.mse(feat_stylized_std, feat_std)

    return loss

def style_loss(feat, feat_stylized):
    return tf.reduce_sum([
        mean_std_loss(f, f_stylized) for f, f_stylized in zip(feat, feat_stylized)
    ])

def content_loss(feat, feat_stylized):
    return tf.reduce_mean(tf.square(feat-feat_stylized), axis=[1,2,3])
