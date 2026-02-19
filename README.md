# Text Mining & Analysis of Academic Journal Data 📝📊

## 📌 Project Overview
Analyzed a dataset of academic journals to extract thematic trends and model publication characteristics. This project focused on transforming unstructured text data (titles, abstracts, and metadata) into structured, actionable insights using Natural Language Processing (NLP) and statistical modeling in **R**.

## 🛠️ Tech Stack & Methodology
* **Language:** R
* **Data Engineering & Text Processing:** Utilized libraries like 'dplyr', 'stringr', and 'tm' (or 'tidytext') to clean text, remove stop-words, apply stemming/lemmatization,and generate Document-Term Matrices (DTM).
* **Text Mining Techniques:**
  * Applied **TF-IDF** (Term Frequency-Inverse Document Frequency) to identify the most significant terms across different journal categories.
  * Conducted **Topic Modeling (LDA)** to discover latent themes within the journal publications.
* **Supervised Learning and Modeling:**
  * Used Multiple Linear Regression Models to derive the impact of factors such as length, age and topic of the journal and predict citation counts.
  * Used Classification methods such as **Logistic Regression** and **Random Forest** to classify abstracts into correctly suited journals.
* **Unsupervised Learning**
  * Performed **Clustering** using the **Partitioning Around Medoids (PAM) algorithm** using the Topic Probabilities (Gammas) derived from the LDA model.
* **Visualization:** Used 'ggplot2' to create word clouds, frequency distributions, and model diagnostic plots.

## 🚀 Key Insights
1. **Thematic Trends:** Discovered that the data is best described by three core themes (AI, Operations Research, and Mathematics) rather than four distinct journal identities.
2. **Citation Counts:** Longer and Older papers expectedly have higher citation counts as well as the fact that papers focusing on AI topics receive more citations than the OR topics.
3. **Model Performance:** The classification and Clustering models successfully categorized abstracts by Topic into 3 distinct domain-based groups **Theoretical Mathematics**, **Operations Research** and **Artificial Intelligence**.

## 📁 Repository Structure
* `/scripts`: R script containing the full NLP pipeline and statistical models (`Journal Data Analysis.R`).
* The comprehensive analytical report detailing the methodology and text mining outputs (`Text Mining & Analysis of Academic Journal Data.pdf`).
* `/data`: *(Note: Dataset withheld or minimized due to size/privacy, but code is fully reproducible).*
  
