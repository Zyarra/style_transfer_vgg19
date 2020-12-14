# style_transfer_vgg19
Style transfer with vgg19 backbone


## Based on https://github.com/xunhuang1995/AdaIN-style
![model_architecture](https://github.com/Zyarra/style_transfer_vgg19/blob/main/images/examples/architecture.jpg)



## Training requires:
1) content images in images/content_images
2) style images in images/style_images
3) output is in images/output_images/stylized.jpg
### and a lot of VRAM/RAM/Time :)

## Parser args:
--content-image(required)
--style-image(required_
--alpha(default 1.0)

Should be something like this:

![content](https://github.com/Zyarra/style_transfer_vgg19/blob/main/images/examples/sailboat_cropped.jpg) + ![style](https://github.com/Zyarra/style_transfer_vgg19/blob/main/images/examples/sketch_cropped.jpg) = ![content_style](https://github.com/Zyarra/style_transfer_vgg19/blob/main/images/examples/sailboat_stylized_sketch.jpg)


