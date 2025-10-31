import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

contracts = pd.read_csv("dataset/contracts.csv")

sns.scatterplot(data=contracts, x="service time", y="value", hue='position', palette='viridis', legend=False)
plt.title('Service Time vs Value by Position')
plt.xlabel('Service Time')
plt.ylabel('Value')
plt.show()