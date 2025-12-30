# Lung and Colon Cancer Histopathological Image Classification

## Project Overview and Purpose
This project implements a Deep Learning solution to classify histopathological images of lung and colon tissues. Using a custom Convolutional Neural Network (CNN), the model distinguishes between five classes: Lung Adenocarcinoma, Lung Squamous Cell Carcinoma, Benign Lung Tissue, Colon Adenocarcinoma, and Benign Colon Tissue. This automation aims to support pathologists by providing a highly accurate second opinion in cancer diagnosis.

## Key Technologies and Libraries
- **Deep Learning**: `TensorFlow`, `Keras`
- **Computer Vision**: `OpenCV` (`cv2`), `PIL`
- **Data Handling**: `NumPy`, `Pandas`
- **Visualization**: `Matplotlib`, `Seaborn`
- **Evaluation**: `Scikit-learn` (Classification Report, Confusion Matrix)

## Dataset and Methodology
### Dataset
The project uses the **Lung and Colon Cancer Histopathological Images** dataset containing 25,000 color images (768 x 768 pixels) equally distributed across five classes.

### Workflow
1. **Data Preprocessing**: 
   - Compiled file paths and labels into a unified Pandas DataFrame.
   - Performed an 80/10/10 split for training, validation, and testing.
2. **Data Augmentation**: Used `ImageDataGenerator` to normalize pixel values and apply real-time augmentations (rotation, zoom, horizontal flip) to improve model robustness.
3. **Model Architecture**:
   - Built a `Sequential` CNN with alternating `Conv2D` and `MaxPooling2D` layers.
   - Integrated `BatchNormalization` for training stability and `Dropout` to prevent overfitting.
   - Final layers include a `Flatten` layer followed by a `Dense` output layer with Softmax activation for 5-class classification.
4. **Optimization**: Compiled using the `Adamax` optimizer and categorical cross-entropy loss.


## Results and Insights
- **Performance**: The model achieves exceptional accuracy on the test set, effectively distinguishing between different cancer types and benign tissues.
- **Clinical Value**: The inclusion of benign tissue classes ensures the model is not just identifying cancer, but accurately identifying the absence of it, which is critical for reducing false positives.
- **Visual Analytics**: The project generates training/validation curves and a confusion matrix to highlight class-specific performance.

## How to Run
1. **Dataset**: Download the dataset from Kaggle (`andrewmvd/lung-and-colon-cancer-histopathological-images`) and update the `data_dir` path in the notebook.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
