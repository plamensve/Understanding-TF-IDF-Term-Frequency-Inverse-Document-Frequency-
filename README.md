# Lyrics Clustering Project

This project performs text preprocessing, vectorization, dimensionality reduction, and clustering on song lyrics. The goal is to discover semantic groups of lyrics and interpret them using top TF-IDF words.

---

## 📌 Steps in the Pipeline

### 1. Load and Prepare the Corpus
- Extract the *Lyrics* column
- Remove missing values
- Convert to a list of strings

### 2. TF-IDF Vectorization
We apply `TfidfVectorizer` with:
- Token pattern: alphabetic words with 4+ letters  
- English stopword removal  
- `max_df=0.1` (remove very frequent words)  
- `min_df=3` (remove very rare words)

This transforms all lyrics into a TF-IDF matrix.

---

## 📉 Normalization
We use `Normalizer()` to scale each TF-IDF vector to unit norm.  
This ensures fair distance comparison between lyrics.

---

## 🔻 Dimensionality Reduction (SVD)
`TruncatedSVD(n_components=2)` compresses high-dimensional TF-IDF vectors into 2 semantic dimensions, making the data easy to visualize and cluster.

---

## 🔍 Clustering (KMeans)
- We cluster the lyrics using **KMeans with 5 clusters**
- We also compute inertia values for k = 2 to 10 to identify the elbow point

---

## 📊 Visualization
A 2D scatter plot shows how lyrics are grouped in the reduced semantic space.

---

## 🧠 Cluster Interpretation
For each cluster:
- Compute average TF-IDF scores
- Extract the top 50 most characteristic words
- Display them in a DataFrame

This helps understand the dominant themes of each cluster.

---

## 📁 Project Output
- `x_svd` → 2D reduced representation  
- `labels` → cluster assignments  
- `top_words_df` → table of top words for each cluster  
- Visualizations of clusters

---

## 🛠 Technologies Used
- Python  
- pandas  
- NumPy  
- scikit-learn  
- matplotlib  

---
