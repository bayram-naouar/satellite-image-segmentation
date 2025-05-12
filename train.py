from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

import yaml
import os

from dataset import SatelliteDataset

def conv_block(x, filters, dropout_rate=0.05):
    """
    Builds a convolutional block consisting of two Conv2D layers, each followed by 
    Batch Normalization, ReLU activation, and Dropout.

    Args:
        x (tf.Tensor): Input tensor to the block.
        filters (int): Number of filters for the Conv2D layers.
        dropout_rate (float): Dropout rate to apply after each activation. Default is 0.1.

    Returns:
        tf.Tensor: Output tensor after applying the convolutional block.
    """
    # First convolution layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.ReLU()(x)

    # Second convolution layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.ReLU()(x)

    return x


def unet(input_size=(128, 128, 3), dropout_rate=0.05):
    """
    Builds the U-Net model.
    
    Args:
        input_size (tuple): The input size of the images. Default is (256, 256, 3).
    
    Returns:
        model: A U-Net model.
    """
    
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = conv_block(inputs, 32)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    # Model
    model = Model(inputs=inputs, outputs=output)

    return model



def bce_dice_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Combines Binary Cross-Entropy (BCE) and Dice Loss.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        alpha: Weight for Binary Cross-Entropy loss.
        beta: Weight for Dice loss.
    
    Returns:
        Total loss (weighted sum of BCE and Dice loss).
    """
    
    # Binary Cross-Entropy Loss
    bce_loss = K.binary_crossentropy(y_true, y_pred)
    
    # Dice Loss
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # Sum over batch and spatial dimensions
    union = K.sum(y_true + y_pred, axis=[1, 2, 3])
    dice_loss = 1 - (2. * intersection + K.epsilon()) / (union + K.epsilon())
    
    # Combine the losses
    total_loss = alpha * bce_loss + beta * dice_loss
    
    return total_loss


# Data parameters config
config = yaml.safe_load(open("config.yaml"))
image_dir = config["data"]["image_dir"]
mask_dir = config["data"]["mask_dir"]
augmented_image_dir = config["data"]["augmented_image_dir"]
augmented_mask_dir = config["data"]["augmented_mask_dir"]
metadata_csv = config["data"]["metadata_csv"]
shuffle = config["data"]["shuffle"]

# Training parameters config
batch_size = config["training"]["batch_size"]
image_size = config["training"]["image_size"]
epochs = config["training"]["epochs"]
use_augment = True
if not os.path.exists(augmented_image_dir) or not os.path.exists(augmented_mask_dir):
    use_augment = False
use_augment = config["training"]["use_augment"]
learning_rate = config["training"]["learning_rate"]
dropout_rate = config["training"]["dropout_rate"]

# Initialize the dataset
dataset = SatelliteDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    augmented_image_dir=augmented_image_dir,
    augmented_mask_dir=augmented_mask_dir,
    metadata_csv=metadata_csv,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=shuffle,
    use_augment=use_augment
)

# Build U-Net model
model = unet(input_size=(*image_size, 3), dropout_rate=dropout_rate)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=bce_dice_loss,
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('models/model_best.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the model
hist = model.fit(dataset,
                 epochs=epochs,
                 callbacks=[checkpoint, early_stopping],
                 verbose=1)

# Save the final model
model.save('models/final_model.h5')

# Plot loss and accuracy
plt.plot(hist.history['loss'], label="Train - Loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.show()
