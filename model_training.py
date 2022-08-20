import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import save_img
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
from variables import *
from utils import *

# Layers to used for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# Layer used for the content loss.
content_layer_name = "block5_conv2"

num_of_content_layer = len(content_layer_name)
num_of_style_layer = len(style_layer_names)

# Using vgg19 to extract features
model_vgg19_conv = vgg19.VGG19(weights="imagenet", include_top=False)
output_layers = dict([(layer.name, layer.output)
                     for layer in model_vgg19_conv.layers])
extractor = tf.keras.Model(
    inputs=model_vgg19_conv.input, outputs=output_layers)

#compute total loss
def compute_loss(generated_image, content_image, style_reference_image):
    input_tensor = tf.concat(
        [content_image, style_reference_image, generated_image], axis=0
    )

    features = extractor(input_tensor)
    loss = tf.zeros(shape=())

    # Content loss
    layer_features = features[content_layer_name]
    content_image_features = layer_features[0, :, :, :]
    generated_image_features = layer_features[2, :, :, :]
    loss = loss + CONTENT_WEIGHT*content_loss(
        content_image_features, generated_image_features
    )/num_of_content_layer

    # Style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_image_features = layer_features[1, :, :, :]
        generated_image_features = layer_features[2, :, :, :]
        E = style_loss(style_image_features, generated_image_features)
        loss = loss + (STYLE_WEIGHT*E)/num_of_style_layer

    # # Total variation loss
    loss = loss + TOTAL_VARIATION_WEIGHT * total_variation_loss(generated_image)
    return loss

# Define gradient and loss of tf function
@tf.function
def compute_loss_and_gradients(generated_image, content_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(generated_image, content_image,
                            style_reference_image)
    grads = tape.gradient(loss, generated_image)
    return loss, grads

# Using Adaptive Momentum Estimation with learning decay to update gradients
optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(
                                        initial_learning_rate=ALPHA, decay_steps=100, decay_rate=0.96
                                        ),
                                    beta_1=0.99, epsilon=1e-1, )


# Input image
content_image = preprocess_images(CONTENT_IMAGE_PATH)
style_reference_image = preprocess_images(STYLE_IMAGE_PATH)
generated_image = tf.Variable(preprocess_images(CONTENT_IMAGE_PATH))

# Train model
for i in range(ITERATIONS+1):
    loss, grads = compute_loss_and_gradients(
        generated_image, content_image, style_reference_image)
    optimizer.apply_gradients([(grads, generated_image)])
    print("Iteration %d: loss = %.2f" % (i, loss))
    if (i % 50 == 0):
        img = deprocess_images(generated_image.numpy())
        filename = PREFIX + "image_at_iterated_%d.png" % i
        tf.keras.preprocessing.image.save_img(filename, img)

#Show final output image
img = tf.keras.preprocessing.image.array_to_img(deprocess_images(generated_image.numpy()))
img.show()