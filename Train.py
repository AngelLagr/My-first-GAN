# Import necessary libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tensorflow.keras.applications import VGG19
from tqdm import tqdm


# ##################### CREATION OF BUILDING BLOCKS ######################

# Define a residual block for the generator
def res_block(ip):
    # First convolutional layer
    res_model = Conv2D(64, (3, 3), padding="same")(ip)
    # Batch normalization
    res_model = BatchNormalization(momentum=0.5)(res_model)
    # Activation with PReLU
    res_model = PReLU(shared_axes=[1, 2])(res_model)

    # Second convolutional layer
    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    # Batch normalization
    res_model = BatchNormalization(momentum=0.5)(res_model)

    # Add the output to the input (residual connection)
    return add([ip, res_model])


# Define an up scaling block (to increase resolution)
def upscale_block(ip):
    # Convolution with 256 filters
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    # Upsampling (doubles the size of the image)
    up_model = UpSampling2D(size=2)(up_model)
    # Activation with PReLU
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    # Return the up scaling block
    return up_model


# ##################### CREATION OF THE GENERATOR ######################

# Define the generator model
def create_gen(gen_ip, num_res_block):
    # Initial convolutional layer
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    # Activation with PReLU
    layers = PReLU(shared_axes=[1, 2])(layers)

    # Temporarily store the first layer
    temp = layers

    # Add residual blocks
    for i in range(num_res_block):
        layers = res_block(layers)

    # Output convolution after residual blocks
    layers = Conv2D(64, (3, 3), padding="same")(layers)
    # Batch normalization
    layers = BatchNormalization(momentum=0.5)(layers)
    # Add skip connection
    layers = add([layers, temp])

    # Upscale the image (twice)
    layers = upscale_block(layers)
    layers = upscale_block(layers)

    # Final convolutional layer to get the HR image
    op = Conv2D(3, (9, 9), padding="same")(layers)

    # Return the generator model
    return Model(inputs=gen_ip, outputs=op)


# ##################### CREATION OF THE DISCRIMINATOR ######################

# Building block for the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    # Convolutional layer
    disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

    # Apply batch normalization if necessary
    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)

    # Activation with LeakyReLU
    disc_model = LeakyReLU(alpha=0.2)(disc_model)

    # Return the block
    return disc_model


# Define the discriminator model
def create_disc(disc_ip):
    # Initial number of filters
    df = 64

    # Add discriminator layers
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)

    # Flatten the result to prepare for the fully connected layer
    d8_5 = Flatten()(d8)
    # Fully connected layer
    d9 = Dense(df*16)(d8_5)
    # Activation with LeakyReLU
    d10 = LeakyReLU(alpha=0.2)(d9)
    # Final output layer with sigmoid activation (to get a probability)
    validity = Dense(1, activation='sigmoid')(d10)

    # Return the discriminator model
    return Model(disc_ip, validity)


# ##################### CREATION OF THE VGG19 MODEL ######################

# Function to build a pre-trained VGG19 model that outputs image features extracted at the third block of the model

def build_vgg(hr_images_shape):
    # Load the pre-trained VGG19 model with ImageNet weights
    vgg_model = VGG19(weights="imagenet", include_top=False, input_shape=hr_images_shape)

    # Extract features from the third convolutional block
    return Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[10].output)


# ##################### COMBINED MODEL ######################

# Create the combined model
def create_comb(gen_model, disc_model, vgg_model, lr_images_ip, hr_images_ip):
    # Generate HR images from LR images
    gen_img = gen_model(lr_images_ip)

    # Extract features from generated images using VGG
    gen_features = vgg_model(gen_img)

    # Set the discriminator to non-trainable for generator updates
    disc_model.trainable = False
    # Calculate the validity of the generated image using the discriminator
    validity = disc_model(gen_img)

    # Return the combined model
    return Model(inputs=[lr_images_ip, hr_images_ip], outputs=[validity, gen_features])


# ##################### DATA PREPARATION ######################

# Load a fixed number of images (5000 here)
n = 5000

# Load and convert the low-resolution images
lr_list = os.listdir("data/lr_images")[:n]
lr_images = []
for img in lr_list:
    img_lr = cv2.imread("data/lr_images/" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)   

# Load high-resolution images
hr_list = os.listdir("data/hr_images")[:n]
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("data/hr_images/" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)   

# Convert to NumPy array
lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

# Normalize pixel values
lr_images = lr_images / 255.
hr_images = hr_images / 255.

# Split data into training and testing sets
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)

# Get shapes of HR and LR images
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

# Define input for the generator
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)


# ##################### MODELS CREATION ######################

# Create the generator model
generator = create_gen(lr_ip, num_res_block=16)

# Create the discriminator model
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Create the VGG19 model
vgg = build_vgg((128, 128, 3))
vgg.trainable = False

# Create the combined GAN model
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)

# Compile the GAN model with binary_crossentropy and MSE losses
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")


# ##################### TRAINING SETUP ######################

# Set batch size to 1 in order to update the gradients after each individual sample
batch_size = 1

# Prepare batches for training from the training data
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
    
# Training loop for a defined number of epochs
epochs = 1000

# Enumerate training over epochs
for e in range(epochs):
    # Labels for fake and real images
    fake_label = np.zeros((batch_size, 1))  # Assign a label of 0 to all fake (generated images)
    real_label = np.ones((batch_size, 1))  # Assign a label of 1 to all real images.

    # Initialize lists to track generator and discriminator losses
    g_losses = []
    d_losses = []
    
    # Enumerate training over batches.
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]  # Fetch a batch of LR images for training
        hr_imgs = train_hr_batches[b]  # Fetch a batch of HR images for training

        # Generate fake HR images using the generator
        fake_imgs = generator.predict_on_batch(lr_imgs)  # Fake images
        
        # Train the discriminator on real and fake images
        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        
        # Set the discriminator to non-trainable for generator updates
        discriminator.trainable = False
        
        # Average the discriminator losses
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) 
        
        # Extract VGG features for the real HR images
        image_features = vgg.predict(hr_imgs)

        # Train the generator using the GAN model (with VGG loss and adversarial loss)
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
        
        # Append the losses for reporting
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    # Convert losses to numpy arrays for easier averaging
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
    # Calculate average losses
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    
    # Report the progress during training.
    print("epoch:", e+1, "g_loss:", g_loss, "d_loss:", d_loss)

    # Save the generator model after every 10 epochs
    if (e+1) % 10 == 0:
        generator.save("gen_e_" + str(e+1) + ".h5")
