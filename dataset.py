import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.utils import Sequence
import os

class SatelliteDataset(Sequence):
    """
    Keras-compatible data generator for satellite image segmentation.

    Supports optional on-the-fly data augmentation using Albumentations,
    as well as loading both original and augmented datasets from disk.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        augmented_image_dir=None,
        augmented_mask_dir=None,
        batch_size=8,
        image_size=(128, 128),
        shuffle=False,
        use_augment=False,
        **kwargs
    ):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Path to the directory containing original input images.
            mask_dir (str): Path to the directory containing original segmentation masks.
            augmented_image_dir (str, optional): Path to augmented images. Used if use_augmented=True.
            augmented_mask_dir (str, optional): Path to augmented masks. Used if use_augmented=True.
            batch_size (int): Number of samples per batch.
            image_size (tuple): Size to resize input and masks to (height, width).
            shuffle (bool): Whether to shuffle data between epochs.
            use_augmented (bool): Whether to include disk-saved augmented data in the dataset.
        """
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmented_image_dir = augmented_image_dir
        self.augmented_mask_dir = augmented_mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.use_augmented = use_augment

        # Load original image and mask filenames
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if not f.startswith('.') and not f.endswith('.gitkeep')
        ])
        self.mask_filenames = sorted([
            f for f in os.listdir(mask_dir)
            if not f.startswith('.') and not f.endswith('.gitkeep')
        ])

        # If use_augmented is enabled, append those filenames as well
        if self.use_augmented and augmented_image_dir and augmented_mask_dir:
            aug_imgs = sorted(os.listdir(augmented_image_dir))
            aug_masks = sorted(os.listdir(augmented_mask_dir))
            self.image_filenames += [os.path.join('aug', fname) for fname in aug_imgs]
            self.mask_filenames += [os.path.join('aug', fname) for fname in aug_masks]
        
        if self.shuffle:
            self.on_epoch_end()  # Shuffle filenames if needed

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Batch index.

        Returns:
            tuple: (X, y) where X is the batch of input images, and y is the corresponding masks.
        """
        # Select batch slice
        batch_image_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_filenames = self.mask_filenames[index * self.batch_size:(index + 1) * self.batch_size]

        # Load and return batch
        X, y = self.__load_batch(batch_image_filenames, batch_mask_filenames)
        return X, y

    def on_epoch_end(self):
        """
        Shuffle data after each epoch if enabled.
        """
        combined = list(zip(self.image_filenames, self.mask_filenames))
        np.random.shuffle(combined)
        self.image_filenames, self.mask_filenames = zip(*combined)

    def __load_batch(self, image_filenames, mask_filenames):
        """
        Load and preprocess a batch of images and masks.

        Args:
            image_filenames (list): List of image filenames for the batch.
            mask_filenames (list): List of mask filenames for the batch.

        Returns:
            tuple: (X, y) arrays ready for training.
        """
        X = []
        y = []

        for img_file, mask_file in zip(image_filenames, mask_filenames):
            try:
                # Handle augmented images stored in subdirectories
                if img_file.startswith('aug') and mask_file.startswith('aug'):
                    img_path = os.path.join(self.augmented_image_dir, os.path.basename(img_file))
                    mask_path = os.path.join(self.augmented_mask_dir, os.path.basename(mask_file))
                else:
                    img_path = os.path.join(self.image_dir, img_file)
                    mask_path = os.path.join(self.mask_dir, mask_file)

                # Load image and mask
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # Validate loading
                if img is None:
                    print(f"\nWarning: Loading {img_path} as PIL Image.")
                    img = Image.open(img_path)
                    img = np.array(img)
                if mask is None:
                    print(f"\nWarning: Loading {mask_path} as PIL Image.")
                    mask = Image.open(mask_path)
                    mask = np.array(mask)

                # Resize image and mask
                img = cv2.resize(img, self.image_size)
                mask = cv2.resize(mask, self.image_size)

                # Ensure 3 channels for image
                if len(img.shape) == 2:  # grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA image
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Normalize image and binarize mask
                img = img.astype(np.float32) / 255.0
                mask = (mask > 127).astype(np.float32)

                # Expand dims to ensure shape (H, W, 1)
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)

                # Final shape check
                if img.shape != (self.image_size[1], self.image_size[0], 3):
                    print(f"Invalid image shape: {img.shape} from {img_path}")
                    continue
                if mask.shape != (self.image_size[1], self.image_size[0], 1):
                    print(f"Invalid mask shape: {mask.shape} from {mask_path}")
                    continue

                X.append(img)
                y.append(mask)
            except Exception as e:
                print(f"Error processing {img_file} or {mask_file}: {e}")
                continue

        # Convert to np.array after ensuring shape consistency
        return np.array(X), np.array(y)
