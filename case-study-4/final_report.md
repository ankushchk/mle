## Abstract

This project analyzes an e-commerce business from two perspectives: customer
purchase behavior and customer review language. RFM features are clustered
using K-Means and DBSCAN, while product reviews are classified with a BiLSTM
baseline and a fine-tuned DistilBERT model.

## 1. Data and preprocessing

Describe the source, sample size, missing values, invalid transactions, review
label construction, and class distribution. Explain why star ratings are only
proxy sentiment labels.

## 2. Customer segmentation

### Method

Explain Recency, Frequency, Monetary, log transformation, standardization,
PCA, K-Means model selection, and DBSCAN parameter selection.

### Results

Insert `outputs/case-study-3/pca_kmeans.png` and the t-SNE figures. Report the
selected K, silhouette score, cluster sizes, and DBSCAN noise rate.

### Personas and actions

| Persona | Evidence from RFM | Recommended action | Success metric |
|---|---|---|---|
| Champions |  |  |  |
| At-risk valuable customers |  |  |  |
| New or occasional buyers |  |  |  |
| Low-engagement customers |  |  |  |

## 3. Sentiment analysis

### Method

Describe the shared data split, BiLSTM architecture, tokenizer, DistilBERT
fine-tuning settings, and class imbalance strategy.

### Results

| Model | Macro-F1 | Weighted-F1 | Training time | Parameters |
|---|---:|---:|---:|---:|
| BiLSTM |  |  |  |  |
| DistilBERT |  |  |  |  |

Discuss the neutral class, confusion matrices, and at least five incorrectly
classified examples.

## 4. Transfer learning and deployment

DistilBERT should generally provide stronger contextual understanding, but it
requires more memory and usually has greater inference latency. Discuss CPU
serving, batching, quantization, knowledge distillation, confidence thresholds,
privacy, monitoring, and model drift.

## 5. Integrated business recommendations

Explain how negative reviews from high-value or at-risk customers can be
prioritized for service recovery, while positive reviews from loyal customers
can support advocacy campaigns.

## 6. Limitations and conclusion

Discuss rating-label noise, clustering instability, DBSCAN sensitivity to
parameters, t-SNE interpretation, sampling bias, and the gap between offline
metrics and production value.
