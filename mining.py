import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def association_rule_mining(data):
    assoc_data = data[['Depression', 'Anxiety', 'Panic Attack', 'Treatment']].astype(bool)
    frequent_itemsets = apriori(assoc_data, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(by='confidence', ascending=False)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

def kmeans_clustering(data, n_clusters=3):
    features = data[['Age', 'Depression', 'Anxiety', 'Panic Attack']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    return data
