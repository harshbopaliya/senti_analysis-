# Sentiment Analysis on Amazon Food Reviews

This project demonstrates a sentiment analysis pipeline applied to Amazon food reviews using two models:
1. **VADER (Valence Aware Dictionary and sEntiment Reasoner):** A rule-based model for quick sentiment scoring.
2. **RoBERTa (Hugging Face):** A state-of-the-art pre-trained transformer model for detailed sentiment classification.

## Project Overview
The aim of this project is to classify Amazon food reviews into **Positive**, **Neutral**, and **Negative** categories, leveraging both VADER and RoBERTa models for comparative analysis and accuracy.

---

## Features
- **Data Preprocessing:** Clean and prepare the Amazon food review dataset.
- **Sentiment Analysis:**
  - Apply the VADER sentiment analysis model for quick scoring.
  - Apply the RoBERTa model using Hugging Face for robust sentiment classification.
- **Result Comparison:** Visualize and compare the sentiment classification outputs from both models.

---

## Tech Stack
- **Python**: Programming language for implementation.
- **Libraries:**
  - `pandas` - For data manipulation.
  - `matplotlib` & `seaborn` - For visualization.
  - `nltk` - For VADER sentiment analysis.
  - `transformers` - For Hugging Face's RoBERTa model.
  - `scikit-learn` - For performance evaluation metrics.
- **Jupyter Notebook**: Used for the entire implementation.

---

## Dataset
- **Source:** [Amazon Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Attributes:**
  - `Review Text`: The actual customer review.
  - `Sentiment`: Ground truth sentiment (if available).

---

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook sentiment_analysis_amazon_reviews.ipynb
   ```

2. Follow the steps outlined in the notebook to:
   - Load and preprocess the dataset.
   - Perform sentiment analysis using VADER.
   - Perform sentiment analysis using RoBERTa.
   - Visualize and compare the results.

---

## Results
- **Metrics Evaluated:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Visualizations:**
  - Distribution of sentiments for VADER and RoBERTa models.
  - Performance comparison plots.

---

## Future Enhancements
- Fine-tune RoBERTa on the Amazon Food Reviews dataset for improved performance.
- Explore other transformer models such as BERT or DistilBERT for comparative analysis.
- Implement a web interface for real-time sentiment analysis.

---

## Contributing
Contributions are welcome! Feel free to suggest improvements or share feedback.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Amazon Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
