# Promotions-Analysis-ML


The 4 datafiles are present in the data directory.

Execute the main file as - python3 main.py

Data will be parsed and loaded from the json file and be in the form of 4 separate dataframes. The goal is to remove all data inconsistencies and concatenate 
the 4 separate dataframes into one main dataframe. Please note that the cleaning steps taken are specific to the promotion at hand. The steps might differ depending 
on the promotion we're trying to analyze.

We also define the "success" metric and add this as a column to our main dataframe. This metric is alterable.

Post creation of the main dataframe, we will separate the features into X variable and y is our target variable (success in this case). We further 
split the X, y in 80:20 ratio for training-testing.

All the models listed in the model directory are ran over the data and a final comparison table is displayed in the terminal. This table gives the performance 
analysis of the models and you can pick which model suits your problem the best. This comparison table is also saved to the Promotions-Model-Evaluation.csv file.

Next up, we move to unsupervised learning. For this use case, we do not employ a "success" metric and use the entire dataset to check if we can find clusters within data.
The main dataframe is used in this case. Please note that the success metric column is dropped. Data features are scaled and following this, two elbow plots are displayed.

We're looking to see at which index on x axis, we can find the most distinct bump (it was found at 2 for the specified promotion). This is an important step as this indicated the 
number of clusters our clustering algorithm will be searching for. Please update the number of clusters found in the parameter n_clusters 
in function def agglomerative_clustering(df_clusters, df) in models/clustering.py

Note- In case more than two clusters are found, you'll have to create their corresponding subset dataframes. More specifically, we'll have to create more subset dataframes like below-
      X_HAC_0 = df[df_clusters["label"] == 0]
      X_HAC_1 = df[df_clusters["label"] == 1]

Post this, for the Promotion ID: 1001460 , two clusters were found, one of which was found to be more perceptive to the promotion.
Its subsequent dataframe was saved to TargetGroup-Cluster0.csv


