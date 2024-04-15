import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Handling missing values if necessary
df = df.dropna()

# Text normalization: Convert to lowercase, remove punctuation, etc.
df['text'] = df['text'].str.lower().str.replace('[^\w\s]', '')

# Vectorization: Transform text data into numerical data
tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Likes', 'RetweetCount']])

# Reducing dimensions of tf-idf features for visualization
pca = PCA(n_components=2)
pca_tfidf_features = pca.fit_transform(tfidf_matrix.toarray())

# Combine PCA reduced tf-idf features with scaled features
combined_features = np.hstack((scaled_features, pca_tfidf_features))

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(combined_features)

# Add the cluster labels to the dataframe
df['cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_tfidf_features[:, 0], y=pca_tfidf_features[:, 1], hue=df['cluster'], palette='viridis')
plt.title('Clusters Visualization')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.show()

# Interpretation of the clusters
for i in range(kmeans.n_clusters):
    cluster_subset = df[df['cluster'] == i]
    
    # Create word clouds for each cluster
    text = ' '.join(cluster_subset['text'].tolist())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {i}')
    plt.show()
    
    # Display the common stats for each cluster
    common_stats = cluster_subset[['Likes', 'RetweetCount']].mean()
    print(f'Cluster {i} average likes: {common_stats["Likes"]}')
    print(f'Cluster {i} average retweets: {common_stats["RetweetCount"]}\n')
