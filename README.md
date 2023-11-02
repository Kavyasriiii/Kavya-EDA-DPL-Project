BLOG ON CROP YIELD ANALYSIS: Unraveling Insights with EDA and Data Preprocessing Introduction. Agriculture is the backbone of the Indian economy, and the productivity of crops is crucial for ensuring food security and economic stability. In the realm of agriculture, crop yield analysis serves as the convergence point of agronomy, meteorology, geospatial science, and data analytics. This amalgamation equips farmers with a robust toolkit to fine-tune crop production. Its role in contemporary agriculture is paramount, extending beyond ensuring food security. It acts as a staunch advocate for sustainable farming practices, ultimately fortifying the economic stability of both individual cultivators and entire nations. Data Description: Before we embark on our journey, it's essential to understand the dataset at hand:

Source:
Our dataset is taken from data.gov.in The dataset includes information on the yield (in Kg./Hectare) of various crops for different Indian states over five years:
• States
• Crops
• Yield (Kg./Hectare) for 2017-18, 2018-19, 2019-20, 2020-21, and 2021-22
The dataset covers a wide range of crops, including rice, wheat, jowar, bajra, maize, pulses, foodgrains, oilseeds, sugarcane, cotton, and jute & mesta. These crops are vital for both food and industrial production in India.

Exploratory Data Analysis (EDA):

EDA is the compass that guides us through the labyrinth of data. Let's begin with the first steps:

python import pandas as pd

Loading the dataset:
df=pd.read_excel('/content/Crop Yield Analysis.xls')

Data Visualization:
Data visualization is our flashlight in the dark cave of data. It helps us see trends, patterns, and anomalies more clearly. Let's light it up with a couple of examples: Python

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

A. Plotting bar plot years = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"] 
rice_production = [2576, 2638, 2722, 2717, 2802]
plt.figure(figsize=(10, 6))
plt.bar(years, rice_production, color='skyblue')
plt.xlabel('Years')
plt.ylabel('Rice Production (Kg./Hectare)')
plt.title('Rice Production Over the Years')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

B. Plotting a Histogram:
Here we are plotting a histogram for the year Yield(Kg./Hectare)-2021-22
plt.hist(df['Yield (Kg./Hectare) - 2021-22'], bins=20)
plt.xlabel('Yield (Kg./Hectare) - 2021-22')
plt.ylabel('Frequency') 
plt.title('Histogram of Yield (2021-22)')
plt.show()

C. Correlation Matrix : 
correlation_matrix = df.corr() 
plt.figure(figsize=(8, 6)) 
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Heatmap of Crop Yields') 
plt.xlabel('Yield Year')
plt.ylabel('Yield Year') plt.show()

D. Plotting a Pair Plot:
sns.pairplot(df)
plt.title('Pair Plot') 
plt.show()

E. Plotting a Scatter Plot :
Plotting a Scatter Plot for the Yield (Kg./Hectare) - 2017-18 VS Yield (Kg./Hectare) - 2021-22 
plt.scatter(df['Yield (Kg./Hectare) - 2017-18'], df['Yield (Kg./Hectare) - 2021-22']) 
plt.xlabel('Yield (2017-18)')
plt.ylabel('Yield (2021-22)') 
plt.title('Scatter Plot of Yields (2017-18 vs. 2021-22)') 
plt.show()


DATA PREPROCESSING : Now that we have illuminated the path through EDA, we can move on to Data Preprocessing. This phase prepares our data for modeling or more in-depth analysis. 

ONE HOT ENCODING: One-hot encoding is a technique used to convert categorical variables into a binary format so that they can be used in machine learning models. To perform one-hot encoding on the "States" and "Crops" columns in this dataset.
import pandas as pd

#Perform one-hot encoding for the 'States' and 'Crops' columns
data_encoded = pd.get_dummies(df, columns=['States', 'Crops'])

#Display the first few rows of the encoded dataset
print(data_encoded.head())

#Save the encoded dataset to a new CSV file
data_encoded.to_csv('encoded_crop_yields.csv', index=False)

MIN MAX SCALER: MinMaxScaler from the scikit-learn library. Min-Max scaling scales the features to a specified range, typically between 0 and 1. 
from sklearn.preprocessing import MinMaxScaler

#Select the columns you want to scale (exclude the categorical columns)
columns_to_scale = [ 'Yield (Kg./Hectare) - 2017-18', 'Yield (Kg./Hectare) - 2018-19', 'Yield (Kg./Hectare) - 2019-20', 'Yield (Kg./Hectare) - 2020-21', 'Yield (Kg./Hectare) - 2021-22' ]

#Initialize the MinMaxScaler
scaler = MinMaxScaler()

#Fit and transform the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

#Display the first few rows of the scaled dataset
print(df.head())

SUMMARY STATISTICS: Here we calculate the summary of the statistics for Yield (Kg./Hectare) - 2017-18. We calculate Mean,Mean,Standard Deviation, Minimum and Maximum
from matplotlib._api import define_aliases

#Calculate summary statistics
mean_yield = df['Yield (Kg./Hectare) - 2017-18'].mean() 
median_yield = df['Yield (Kg./Hectare) - 2017-18'].median() 
std_deviation = df['Yield (Kg./Hectare) - 2017-18'].std() 
min_yield = df['Yield (Kg./Hectare) - 2017-18'].min() 
max_yield = df['Yield (Kg./Hectare) - 2017-18'].max()

print(f"Mean: {mean_yield:.2f}") 
print(f"Median: {median_yield:.2f}") 
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"Min: {min_yield:.2f}")
print(f"Max: {max_yield:.2f}")

We calculate the summary of the statistics for Yield (Kg./Hectare) - 2021-22. We calculate Mean,Mean,Standard Deviation, Minimum and Maximum 
from matplotlib._api import define_aliases

Calculate summary statistics
mean_yield = df['Yield (Kg./Hectare) - 2017-18'].mean() 
median_yield = df['Yield (Kg./Hectare) - 2017-18'].median() 
std_deviation = df['Yield (Kg./Hectare) - 2017-18'].std() 
min_yield = df['Yield (Kg./Hectare) - 2017-18'].min()
max_yield = df['Yield (Kg./Hectare) - 2017-18'].max()

print(f"Mean: {mean_yield:.2f}")
print(f"Median: {median_yield:.2f}")
print(f"Standard Deviation: {std_deviation:.2f}") 
print(f"Min: {min_yield:.2f}")
print(f"Max: {max_yield:.2f}")

. Justification of this Dataset : This dataset is crucial for several reasons:
• Agricultural Policy: Policymakers can use this data to assess the effectiveness of agricultural policies, such as subsidies, crop insurance, and minimum support prices.
• Crop Selection: Farmers can make informed decisions about which crops to cultivate based on yield trends and market demand. 
• Resource Allocation: Knowing which states and crops are performing well can help allocate resources more efficiently, including water, fertilizers, and agricultural extension services. 
• Food Security: Understanding yield trends is essential for ensuring that India can produce enough food to meet its growing population's needs. 
• Economic Impact: Crop yield trends can have a significant impact on the agricultural sector and the overall economy, as it affects the income of millions of farmers and the agri-industry.

CONCLUSION: Exploratory Data Analysis and Data Preprocessing are the pillars upon which insightful data analysis is built. By meticulously creating informative visualizations, and preparing it for modeling or deeper analysis, we uncover the true potential hidden within the data. In this journey through an Crop Yield Analysis, we've touched upon the essence of EDA and DPL.
