
# Airbnb Listing Price and Type Prediction

This project focuses on a multimodal and multi-task machine learning problem to predict the `type` and `price` category of an Airbnb listing using its textual `summary` and `image`. The price is a categorical variable with three classes (0, 1, 2).

## ✔️ Problem Formulation:

The primary objective is to build a model that can accurately classify an Airbnb listing into one of the predefined `type` categories and one of the three `price` brackets.

-   **Multimodal Input:** The model processes two distinct types of data:
    1.  **Textual Data:** A summary overview of the listing.
    2.  **Image Data:** A representative image of the listing.
-   **Multi-task Output:** The model is trained to predict two different target variables simultaneously:
    1.  **Listing Type:** The category of the property (e.g., Apartment, House).
    2.  **Price Category:** A discretized price range.

To handle these different data types, we employ a combination of Convolutional Neural Networks (CNNs) for image feature extraction and sequential modeling techniques for text analysis.

## 📂 Dataset

The dataset is sourced from the "cisc-873-dm-f22-a4" Kaggle competition. It contains a training set with listing summaries, image paths, types, and price categories, along with a test set for which predictions need to be made.

## 🛠️ Methodology

### 1. Data Loading and Inspection

-   The training and testing data are loaded from CSV files using the `pandas` library.
-   Initial data investigation includes checking the shape of the data, looking for missing values (`NaNs`), and identifying any duplicate entries.

### 2. Data Preprocessing

A two-fold preprocessing strategy is adopted to handle the multimodal inputs:

**Image Data:**
-   Images are loaded using the `Pillow` library.
-   Each image is converted to a two-channel format (Luminance and Alpha).
-   All images are resized to a uniform dimension of `64x64` pixels to ensure consistency for the neural network input.

**Text Data:**
-   The `type` column is label-encoded into numerical categories.
-   A `Tokenizer` from `tensorflow.keras.preprocessing.text` is used to build a vocabulary from the training summaries.
-   The text is then converted into sequences of integers.
-   `pad_sequences` is applied to ensure all text sequences have a uniform length (`max_len = 100`).

### 3. Model Architecture

A multi-input, multi-output Keras functional API model is constructed to handle this task.

1.  **Text Input Branch:**
    -   An `Embedding` layer converts the integer sequences into dense vectors of size 100.
    -   A `Lambda` layer calculates the mean of these embeddings to get a single feature vector for the summary.

2.  **Image Input Branch:**
    -   A `Conv2D` layer with 64 filters of size 16x16 is used to extract features from the images.
    -   A `MaxPool2D` layer downsamples the feature maps.
    -   A `Flatten` layer converts the 2D feature maps into a 1D vector.

3.  **Fusion and Output:**
    -   The feature vectors from the text and image branches are concatenated.
    -   The fused layer is then connected to two separate `Dense` output layers:
        -   One for `type` prediction with a `softmax` activation function.
        -   One for `price` prediction with a `softmax` activation function.

### 4. Model Compilation and Training

-   **Optimizer:** The `Adam` optimizer is used.
-   **Loss Function:** `sparse_categorical_crossentropy` is used for both outputs since they are multi-class classification tasks.
-   **Loss Weights:** Both tasks are given equal importance with a weight of `0.5`.
-   **Metrics:** `SparseCategoricalAccuracy` is tracked for both outputs.
-   **Training:** The model is trained for 20 epochs with a batch size of 8. An `EarlyStopping` callback is used to prevent overfitting by monitoring the validation loss on the price prediction.

### 5. Prediction and Submission

-   The trained model is used to predict the `type` and `price` for the preprocessed test data.
-   The output probabilities are converted into class predictions using `np.argmax`.
-   A submission file is generated in the required format containing the `id` and the predicted `price` category.


## 🚀 How to Run the Code

1.  **Setup Kaggle API:**
    -   Create a `kaggle.json` API token from your Kaggle account.
    -   Upload the `kaggle.json` file to the environment.

2.  **Install Dependencies:**
    -   Ensure you have `pandas`, `numpy`, `tensorflow`, `matplotlib`, and `scikit-learn` installed.

3.  **Execute the Notebook:**
    -   Run the cells in the provided notebook sequentially to download the data, preprocess it, build the model, train it, and generate the final submission file.
