# 🛰️ Satellite Image Segmentation with U-Net

A complete deep learning pipeline for segmenting satellite images using a U-Net model, including data augmentation, training, and an interactive Streamlit playground.

---

## 📁 Project Structure

```
├── data/
│   ├── Image/                  # Original input images
│   ├── Mask/                   # Corresponding masks
│   ├── augmented_images/       # Augmented input images
│   └── augmented_masks/        # Augmented masks
├── models/
│   ├── final_model.h5          # Final saved model
│   └── model_best.h5           # Best checkpoint model
├── app.py                      # Streamlit app for testing images
├── dataset.py                  # Dataset class and preprocessing logic
├── augment_data.py             # Script to augment images and masks
├── train.py                    # Model training and loss plotting
├── config.yaml                 # Configuration file for parameters
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files to ignore in Git
```

---

## ⚙️ How to Use

1. **(Optional)** Augment the dataset:
```bash
python augment_data.py
```

2. **Train the model**:
```bash
python train.py
```

3. **Run the Streamlit app**:
```bash
streamlit run app.py
```

---

## 🧪 Features

- U-Net architecture with optional Dropout and BatchNormalization
- Combined Binary Cross-Entropy and Dice Loss
- Real-time inference with Streamlit
- YAML-based config for easy experimentation

---

## 🔧 Configuration

- All important settings for data paths, training behavior, and model architecture are centralized in the config.yaml file.  
- This allows you to easily customize the project without modifying the core scripts.  
- You can change dataset directories (```image_dir```, ```mask_dir```, ```augmented_image_dir```, ```augmented_mask_dir```) or enable/disable data shuffling.  
- Under the training section, you can tune hyperparameters such as ```batch_size```, ```image_size```, ```number of epochs```, ```learning_rate```, and ```dropout_rate```.  
- If you want to include augmented data during training, simply set ```use_augment``` to ```true```.  
- Finally, the model section lets you specify which architecture to use (currently supports "unet").  
- Adjust these values as needed to experiment with performance or scale the pipeline to different datasets.  

---

## 📦 Setup

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## ✉️ Contact

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/bayramnaouar95) or via email: bayram.naouar@gmail.com
