Fake News Classifier using Neural Networks

A Fake News Classifier is a system designed to automatically identify and categorize news articles or headlines as either genuine or fake. The primary goal is to distinguish between trustworthy information and misinformation, aiding in the battle against the spread of false narratives.

Fake news poses a significant threat to information integrity and public perception. The ability to automatically detect and categorize fake news helps in preventing the spread of misleading information, fostering media literacy, and promoting a more trustworthy digital environment.

In this project I aimed to develop a robust Fake News Classifier using a neural network model. By leveraging natural language processing and machine learning techniques, the classifier can analyze textual content and make predictions regarding the authenticity of news articles.

Libraries used:

1) NLTK (Natural Language Toolkit): Used for comprehensive natural language processing 

NLTK played a crucial role in text cleaning and preprocessing. It facilitated tasks such as stemming, where words were reduced to their root form, promoting uniformity in the representation of words. Additionally, NLTK's stopwords module was employed to remove common English words that typically do not contribute much information to the meaning of a sentence.


2) Pandas: Essential for data manipulation and analysis

It was utilized to load the dataset containing news articles, drop any rows with missing values, and reset the DataFrame index. This ensured a clean and structured dataset for subsequent processing.


3) NumPy: An integral library for numerical operations in Python

NumPy was employed to efficiently handle numerical arrays. The one-hot encoded representations of the text data were converted to NumPy arrays, providing a streamlined and efficient structure for subsequent processing in the neural network model.

4) TensorFlow: Served as the backend for neural network operations

TensorFlow provided the computational foundation for building and training deep learning models. It efficiently handled the low-level operations required for neural network training, allowing for seamless execution of complex computations on hardware accelerators.

5) Keras: A high-level neural network API that runs on top of TensorFlow

Keras simplified the process of designing and training neural networks by offering a user-friendly interface. In this project, the Sequential model from Keras was used to sequentially stack layers for building the architecture of the neural network. 

6) Scikit-Learn: A versatile machine learning library that includes various tools for data preprocessing and model evaluation. 

Scikit-Learn played a pivotal role in several aspects:
a) Train-Test Splitting: The `train_test_split` function from Scikit-Learn was employed to split the dataset into training and testing sets, facilitating the evaluation of the model's performance on unseen data.

b) Performance Evaluation: Scikit-Learn's metrics, such as the confusion matrix and accuracy score, were used to quantitatively assess the model's classification performance. These metrics provided insights into the model's ability to correctly classify instances of genuine and fake news.

Process Steps:

   a. Data Loading and Preprocessing: Loads a dataset containing news articles, cleans the text, and prepares it for model training.
   
   b. Text Preprocessing: Utilizes NLTK for stemming and stopwords removal to enhance the quality of textual data.
   
   c. One-Hot Encoding and Padding: Employs Keras functions to convert text data into numeric vectors and ensures uniform length for input sequences.
   
   d. Model Definition: Constructs a neural network model using the Keras Sequential API with layers like Embedding, LSTM, Dropout, and Dense for binary classification.
   
   e. Model Training: The model is trained on a training set, with performance monitored on a validation set to ensure generalizability.
   
   f. Evaluation: Utilizes Scikit-Learn to assess the model's performance using a confusion matrix and accuracy score.

Neural Network Model for Text Classification:

The neural network model employs a sequential architecture with several key components:
   
1) Embedding Layer: Converts words into dense vectors, capturing semantic relationships.

2) LSTM Layer: Long Short-Term Memory layer for understanding context and sequential dependencies in the input data.

3) Dropout Layers: Introduce regularization by randomly dropping neurons during training, preventing overfitting.

4) Dense Layer: The final layer with a single neuron and a sigmoid activation function for binary classification.

Keras API Sequential Architecture:

In Keras, the Sequential model allows for the linear stacking of layers, making it simple to create a neural network step by step.

1) Embedding Layer: Represents word vectors in a continuous space, capturing semantic relationships between words.

2) Dropout Layers: Randomly drop a fraction of input units during training to prevent overfitting and improve model generalization.

3) LSTM Layer: A type of recurrent neural network layer that is well-suited for understanding sequential data by maintaining a memory of previous timesteps.

4) Dense Layer: The final layer with a single neuron and a sigmoid activation function, making it suitable for binary classification tasks by outputting probabilities.

Possible Applications:

Social Media Content Moderation
News Aggregator Systems

Possible Improvements:

Incorporating pre-trained word embeddings (e.g., Word2Vec, GloVe) for richer semantic understanding.
Experimenting with different neural network architectures and hyperparameter tuning.
Continuous model retraining to adapt to evolving language patterns.

Understanding the output:

The confusion matrix provides a detailed breakdown of the performance of a classification model. In this case, the confusion matrix is a 2x2 matrix.

[[3118, 301],
 [231,  2385]]

Understanding the confusion matrix:

- True Positives (TP): 2385
  - These are cases where the actual class is positive (e.g., genuine news), and the model correctly predicted them as positive.

- True Negatives (TN): 3118
  - These are cases where the actual class is negative (e.g., fake news), and the model correctly predicted them as negative.

- False Positives (FP): 301
  - These are cases where the actual class is negative, but the model incorrectly predicted them as positive. Also known as Type I error.

- False Negatives (FN): 231
  - These are cases where the actual class is positive, but the model incorrectly predicted them as negative. Also known as Type II error.

In the context of the fake news classifier:

- The model correctly identified 2385 instances of genuine news (True Positives).
- It correctly identified 3118 instances of fake news (True Negatives).
- It misclassified 301 instances of fake news as genuine (False Positives).
- It missed 231 instances of genuine news, classifying them as fake (False Negatives).

Interpretation:
- The high True Positives (2385) and True Negatives (3118) indicate that the model is effective at correctly classifying both genuine and fake news.
- The False Positives (301) suggest instances where the model incorrectly identified fake news as genuine, potentially leading to misinformation.
- The False Negatives (231) indicate instances where the model missed genuine news, potentially causing a lack of information.

Metrics:
- Accuracy: (TP + TN) / (TP + TN + FP + FN) = (2385 + 3118) / (2385 + 3118 + 301 + 231) ≈ 0.922
  - The accuracy of the model is approximately 92.2%, which is the ratio of correctly predicted instances to the total instances.

- Precision: TP / (TP + FP) = 2385 / (2385 + 301) ≈ 0.888
  - Precision is the ratio of correctly predicted positive observations to the total predicted positives. In this case, it's approximately 88.8%.

- Recall (Sensitivity): TP / (TP + FN) = 2385 / (2385 + 231) ≈ 0.911
  - Recall is the ratio of correctly predicted positive observations to all the actual positives. In this case, it's approximately 91.1%.

- F1 Score: 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.888 * 0.911) / (0.888 + 0.911) ≈ 0.899
  - The F1 Score is the weighted average of Precision and Recall. In this case, it's approximately 89.9%.


