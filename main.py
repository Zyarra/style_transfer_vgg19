import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
from utilities import load_img, resize_img, style_loss, content_loss
from models import TransferNet, VGG

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(f'GPU is available: {tf.test.is_gpu_available()}')

AUTOTUNE = tf.data.experimental.AUTOTUNE

log_dir = 'model'
style_dir = 'images/style_images'
content_dir = 'images/content_images'
learning_rate = 1e-4
lr_decay = 5e-5
max_steps = 160000
image_size = 224
batch_size = 8
content_weight = 1
style_weight = 5
log_freq = 500

content_paths = ["avril_cropped.jpg", "chicago_cropped.jpg"]
style_paths = ["impronte_d_artista_cropped.jpg", "ashville_cropped.jpg"]

test_content_images = tf.concat([load_img(f"images/content_images/{f}") for f in content_paths], axis=0)
test_style_images = tf.concat([load_img(f"images/style_images/{f}") for f in style_paths], axis=0)

content_layer = "block4_conv1"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1"]

vgg = VGG(content_layer, style_layers)
transformer = TransferNet(content_layer)
vgg(test_style_images)


def resize_and_crop(img, min_size):
    img = resize_img(img, min_size=min_size)
    img = tf.image.random_crop(img, size=(image_size, image_size, 3))
    img = tf.cast(img, tf.bfloat16)
    return img


def process_content(features):
    img = features["image"]
    img = resize_and_crop(img, min_size=286)
    return img


def process_style(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = resize_and_crop(img, min_size=512)
    return img


dset = tf.data.Dataset.list_files(style_dir + '/*.jpg')
dset = dset.map(process_style, num_parallel_calls=AUTOTUNE)
dset = dset.apply(tf.data.experimental.ignore_errors())
dset = dset.repeat().batch(batch_size=batch_size).prefetch(AUTOTUNE)

ds_c = tf.data.Dataset.list_files(content_dir + '/*.jpg')
ds_c = ds_c.map(process_style, num_parallel_calls=AUTOTUNE)
ds_c = ds_c.apply(tf.data.experimental.ignore_errors())
ds_c = ds_c.repeat().batch(batch_size=batch_size).prefetch(AUTOTUNE)

ds = tf.data.Dataset.zip((ds_c, dset))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, transformer=transformer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=4)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_style_loss = tf.keras.metrics.Mean(name='train_style_loss')
train_content_loss = tf.keras.metrics.Mean(name='train_content_loss')

summary_writer = tf.summary.create_file_writer(log_dir)

@tf.function
def train_step(content_img, style_img):
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print(f'Restored from {checkpoint_manager.latest_checkpoint}')
    else:
        print('No checkpoints found. Initializing...')

    t = transformer.encode(content_img, style_img, alpha=1.0)

    with tf.GradientTape() as tape:
        stylized_img = transformer.decode(t)

        _, style_feat_style = vgg(style_img)
        content_feat_stylized, style_feat_stylized = vgg(stylized_img)

        total_style_loss = style_weight * style_loss(style_feat_style, style_feat_stylized)
        total_content_loss = content_weight * content_loss(t, content_feat_stylized)
        total_loss = total_content_loss + total_style_loss

    gradients = tape.gradient(total_loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(total_loss)
    train_style_loss(total_style_loss)
    train_content_loss(total_content_loss)


for step, (content_images, style_images) in enumerate(ds):
    new_lr = learning_rate / (1.0 + lr_decay * step)
    optimizer.learning_rate.assign(new_lr)
    train_step(content_images, style_images)

    if step % log_freq == 0:
        with summary_writer.as_default():
            tf.summary.scalar("loss/total", train_loss.result(), step=step)
            tf.summary.scalar("loss/style", train_style_loss.result(), step=step)
            tf.summary.scalar("loss/content", train_content_loss.result(), step=step)
            tf.summary.image('stylized', style_images / 255.0, step=step, max_outputs=6)

        print(
            f'Step: {step}, loss: {train_loss.result()}, style_loss: {train_style_loss.result()},'
            f' content_loss: {train_content_loss.result()}')
        print(f'Checkpoint saved: {checkpoint_manager.save()}')

        train_loss.reset_states()
        train_style_loss.reset_states()
        train_content_loss.reset_states()
