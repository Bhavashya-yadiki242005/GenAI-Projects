project_name: News Article Categorizer

level: Foundation (NLP & Machine Learning)

overview: >
  News articles are published daily under categories such as politics,
  business, sports, technology, and entertainment. Manually organizing
  these articles is time-consuming. This project builds an automated
  news article categorization system using Natural Language Processing
  (NLP) and Machine Learning.

purpose:
  - Automate news classification
  - Reduce manual effort in content organization
  - Build strong NLP fundamentals for future GenAI projects

use_cases:
  - Media monitoring for specific topics
  - Content recommendation based on user interests
  - Sentiment and trend analysis on news articles

dataset:
  name: BBC News Dataset
  categories:
    - business
    - politics
    - sport
    - tech
    - entertainment

approach:
  text_preprocessing:
    steps:
      - Convert text to lowercase
      - Tokenize text into words
      - Remove stopwords
      - Remove non-alphabetic characters
    purpose: >
      Cleaning the text helps the model focus only on meaningful words.

  text_representation:
    techniques:
      - name: Bag of Words (BoW)
        description: Converts text into word frequency vectors
      - name: TF-IDF
        description: >
          Assigns importance to words based on how unique they are
          across documents

  model_training:
    algorithm: Multinomial Naive Bayes
    data_split:
      train: 80%
      test: 20%
    description: >
      Separate models are trained using BoW and TF-IDF features
      and evaluated on unseen data.

results:
  performance:
    bow_accuracy: 98%
    tfidf_accuracy: 97%
  observation: >
    Both models perform well across all categories, with the BoW model
    slightly outperforming TF-IDF on this dataset.

custom_prediction:
  example_input: >
    Artificial intelligence is revolutionizing the tech industry,
    with companies racing to develop the next big innovation.
  predicted_output:
    category: tech

technologies_used:
  - Python
  - Pandas
  - NumPy
  - NLTK
  - Scikit-learn
  - Jupyter Notebook
