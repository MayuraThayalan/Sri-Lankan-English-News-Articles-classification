📰 News Article Classifier

A machine learning pipeline that classifies news articles into categories using text from headings and bodies. The trained model is then showed via a Streamlit web app.

📁 Project Structure

- `news heading.csv` – CSV file containing news headings and their categories.
- `news body.csv` – CSV file containing the full news article bodies.
- `main.ipynb` – Jupyter notebook that:
  - Loads and merges heading and body data.
  - Cleans text with regex.
  - Vectorizes text using TF-IDF.
  - Trains a Logistic Regression classifier.
  - Saves the model, vectorizer, and category labels with `pickle`.
- `classifier.pkl` – Trained classifier model.
- `vectorizer.pkl` – Fitted TF-IDF vectorizer.
- `categories.pkl` – List of unique category labels.
- `app.py` – Streamlit app that loads the saved models and lets users classify new news articles.

🛠️ Requirements

To run the training pipeline and app, install:

pip install pandas numpy scikit-learn streamlit
