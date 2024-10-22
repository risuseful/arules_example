# credit: chatgpt

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Generate tab1 with random float values between 0 and 100
np.random.seed(42)  # for reproducibility
tab1 = pd.DataFrame(np.random.uniform(0, 100, size=(100, 5)), columns=['A', 'B', 'C', 'D', 'E'])

# Step 2: Compute the means of each column
col_means = tab1.mean()

# Step 3: Generate tab1_cat (categorical version of tab1)
tab1_cat = tab1.apply(lambda col: np.where(col < col_means[col.name], 'LOW', 'HIGH'))

# Step 4: Prepare data for association rule mining (convert to one-hot encoding)
one_hot_tab1_cat = pd.get_dummies(tab1_cat)

# Step 5: Use Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_tab1_cat, min_support=0.1, use_colnames=True)

# Step 6: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Step 7: Show the association grouping matrix
association_matrix = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(association_matrix.head())


import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create a new dataframe with antecedents, consequents, and the metric (e.g., lift)
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Step 2: Pivot the dataframe to create a matrix (e.g., for lift)
matrix = rules.pivot(index='antecedents', columns='consequents', values='lift')

# Step 3: Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=False, cmap="YlGnBu", linewidths=0.5)
plt.title('Association Rules - Lift Heatmap')
plt.show()
