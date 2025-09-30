import pandas as pd
import matplotlib.pyplot as plt

SHOPPING_DATA_URL = 'Lab1/Part2/shopping_data.csv'
customer_data = pd.read_csv(SHOPPING_DATA_URL)

customer_data = customer_data.drop(['CustomerID', 'Genre', 'Age'], axis=1)

data = customer_data.iloc[:,0:2].values

print(data)

labels = range(1,201)
plt.figure(figsize=(10,7))
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:,0], data[:,1], label='True Position')
plt.show()

"""
    Create clusters
"""