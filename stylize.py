import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(f'GPU is available: {tf.test.is_gpu_available()}')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
from models import TransferNet
from utilities import load_img
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--content-image", required=True)
    parser.add_argument("--style-image", required=True)
    parser.add_argument("--alpha", default=1.0, type=float)
    args = parser.parse_args()


    content_img = load_img(args.content_image)
    style_img = load_img(args.style_image)
    content_layer = "block4_conv1"
    log_dir = 'model'
    output_img_dir = 'images/output_images'

    transformer = TransferNet(content_layer)
    checkpnt = tf.train.Checkpoint(transformer=transformer)
    checkpnt.restore(tf.train.latest_checkpoint(log_dir)).expect_partial()


    stylized_image = transformer(content_img, style_img, alpha=args.alpha)
    stylized_image = tf.cast(tf.squeeze(stylized_image), tf.uint8).numpy()
    img = Image.fromarray(stylized_image, mode="RGB")
    img.save(f'{output_img_dir}/stylized.jpg')

