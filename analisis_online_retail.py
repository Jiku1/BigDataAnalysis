"""
Offline Big Data Analysis - Online Retail (UCI)
Author: Alex Nugraha Setia (241012000064)
Environment: Conda (Python 3.10)
"""

# === 1. IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # <-- Tambahan untuk evaluasi klaster
from mlxtend.frequent_patterns import apriori, association_rules

# === 2. LOAD DATA ===
print("Loading dataset...")
df = pd.read_excel("Data/OnlineRetail.xlsx")

print("Dataset loaded!")
print(f"Shape: {df.shape}")
print(df.head())

# === 3. DATA CLEANING ===
print("\n Cleaning data...")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df.drop_duplicates(inplace=True)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"Data cleaned. Remaining rows: {len(df)}")

# === 4. EXPLORATORY DATA ANALYSIS ===
print("\n Generating EDA plots...")
if not os.path.exists("output"):
    os.makedirs("output")

# Top 10 products
plt.figure(figsize=(10,5))
top_products = df['Description'].value_counts().head(10)
sns.barplot(y=top_products.index, x=top_products.values)
plt.title("Top 10 Most Purchased Products")
plt.tight_layout()
plt.savefig("output/top_products.png")

# Top 10 countries
plt.figure(figsize=(8,5))
top_countries = df['Country'].value_counts().head(10)
sns.barplot(y=top_countries.index, x=top_countries.values)
plt.title("Top 10 Countries by Transaction Count")
plt.tight_layout()
plt.savefig("output/top_countries.png")

print("EDA plots saved in 'output/' folder.")

# === 5. RFM ANALYSIS ===
print("\n Performing RFM analysis...")
now = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (now - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency',
                   'InvoiceNo': 'Frequency',
                   'TotalPrice': 'Monetary'})

# Normalize RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# === 6. K-MEANS CLUSTERING ===
print("\n Running K-Means clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# === 6.1 EVALUASI DENGAN SILHOUETTE SCORE ===
sil_score = silhouette_score(rfm_scaled, rfm['Cluster'])
print(f"\nSilhouette Score: {sil_score:.4f}")

if sil_score > 0.7:
    print("Interpretasi: Klaster sangat terpisah dan jelas.")
elif sil_score > 0.5:
    print("Interpretasi: Klaster cukup baik, ada sedikit tumpang tindih.")
elif sil_score > 0.25:
    print("Interpretasi: Klaster lemah, ada beberapa overlap antar grup.")
else:
    print("Interpretasi: Klaster kurang baik, perlu penyesuaian jumlah cluster.")

# === 6.2 VISUALISASI KLASTER ===
plt.figure(figsize=(8,5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
plt.title("Customer Segmentation (RFM + KMeans)")
plt.tight_layout()
plt.savefig("output/customer_segmentation.png")

print("Clustering complete. Saved plot to 'output/customer_segmentation.png'.")

# === 7. MARKET BASKET ANALYSIS ===
print("\n Performing Market Basket Analysis...")
basket = (df[df['Country'] == "United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.applymap(encode_units)

frequent_items = apriori(basket_sets, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

print("\n Top 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Save to CSV
rules.to_csv("output/association_rules.csv", index=False)
print("Rules saved to 'output/association_rules.csv'")

# === 8. VISUALISASI TOP 10 RULES ===
print("\n Generating Top 10 Association Rules by Confidence...")
top10_rules = rules.sort_values('confidence', ascending=False).head(10)
rule_labels = [f"{list(a)} → {list(c)}" for a, c in zip(top10_rules['antecedents'], top10_rules['consequents'])]

plt.figure(figsize=(10,6))
sns.barplot(x=top10_rules['confidence'], y=rule_labels, palette='coolwarm')
plt.xlabel("Confidence")
plt.title("Top 10 Association Rules by Confidence")
plt.tight_layout()
plt.savefig("output/top10_rules_confidence.png")

print("Top 10 rules plot saved to 'output/top10_rules_confidence.png'")
plt.show()

print("\n✅ Analysis Complete. All results saved in 'output/' folder.")
