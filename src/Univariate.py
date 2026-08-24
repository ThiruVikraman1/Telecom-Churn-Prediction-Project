import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

class univariate():

    def QuanQual(dataset):
        Qual = []
        Quan = []
        for ColumnName in dataset.columns:
            if dataset[ColumnName].dtype == "O":
                Qual.append(ColumnName)
            else:
                Quan.append(ColumnName)
        return Qual,Quan
        
    # Frequency Function
    def Frequency(ColumnName,dataset):
        freqTable = pd.DataFrame(columns = ["Unique Values","Frequency","Relative Frequency","Cusum"])
        freqTable["Unique Values"] = dataset[ColumnName].value_counts().index
        freqTable["Frequency"] = dataset[ColumnName].value_counts().values
        freqTable["Relative Frequency"] = (freqTable["Frequency"]/len(dataset[ColumnName].value_counts()))
        freqTable["Cusum"] = freqTable["Relative Frequency"].cumsum()
        return freqTable

    # Descriptive Function
    def Descriptive_Table(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5_Rule",
                                    "Lesser_range","Greater_range","Min","Max"],columns=quan)
        for ColumnName in quan:
            descriptive[ColumnName]["Mean"] = dataset[ColumnName].mean()
            descriptive[ColumnName]["Median"] = dataset[ColumnName].median()
            descriptive[ColumnName]["Mode"] = dataset[ColumnName].mode()[0]
            descriptive[ColumnName]["Q1:25%"] = dataset.describe()[ColumnName]["25%"]
            descriptive[ColumnName]["Q2:50%"] = dataset.describe()[ColumnName]["50%"]
            descriptive[ColumnName]["Q3:75%"] = dataset.describe()[ColumnName]["75%"]
            descriptive[ColumnName]["99%"] = np.percentile(dataset[ColumnName],99)
            descriptive[ColumnName]["Q4:100%"] = dataset.describe()[ColumnName]["max"]
            descriptive[ColumnName]["IQR"] = descriptive[ColumnName]["Q3:75%"]-descriptive[ColumnName]["Q1:25%"]
            descriptive[ColumnName]["1.5_Rule"] = 1.5* descriptive[ColumnName]["IQR"]
            descriptive[ColumnName]["Lesser_range"] = descriptive[ColumnName]["Q1:25%"]-descriptive[ColumnName]["1.5_Rule"]
            descriptive[ColumnName]["Greater_range"] = descriptive[ColumnName]["Q3:75%"]+descriptive[ColumnName]["1.5_Rule"]
            descriptive[ColumnName]["Min"] = dataset[ColumnName].min()
            descriptive[ColumnName]["Max"] = dataset[ColumnName].max()
            descriptive[ColumnName]["Skew"] = dataset[ColumnName].skew()
            descriptive[ColumnName]["Kurtosis"] = dataset[ColumnName].kurtosis()
            descriptive[ColumnName]["Variance"] = dataset[ColumnName].var()
            descriptive[ColumnName]["Std"] = dataset[ColumnName].std()
        return descriptive

    # Outliers function
    def Outlier_Columns(descriptive,quan):
        Lesser_outlier = []
        Greater_outlier = []
        for ColumnName in quan:
            if descriptive[ColumnName]["Min"]<descriptive[ColumnName]["Lesser_range"]:
                Lesser_outlier.append(ColumnName)
            if descriptive[ColumnName]["Max"]>descriptive[ColumnName]["Greater_range"]:
                Greater_outlier.append(ColumnName)
        return Lesser_outlier,Greater_outlier

    # Outlier handling function
    def Replace_Outliers(Lesser_outlier,Greater_outlier,descriptive,dataset):
        for ColumnName in Lesser_outlier:
            dataset[ColumnName][dataset[ColumnName]<descriptive[ColumnName]["Lesser_range"]]=descriptive[ColumnName]["Lesser_range"]
        for ColumnName in Greater_outlier:
            dataset[ColumnName][dataset[ColumnName]>descriptive[ColumnName]["Greater_range"]]=descriptive[ColumnName]["Greater_range"]

    def get_Pdf_Probablity(dataset,startrange,endrange):
        from matplotlib import pyplot
        from scipy.stats import norm 
        import seaborn as sns
        ax = sns.distplot(dataset,kde=True,kde_kws={"color":"Blue"},color="Green")

        pyplot.axvline(startrange,color="Red")
        pyplot.axvline(endrange,color="Red")

        # Calculate the parameters
        sample = dataset
        sample_mean = sample.mean()
        sample_std = sample.std()
        print(f"mean={sample_mean:.3f}, standard deviation= :{sample_std:.3f}")
        # Normal distribution 
        dist = norm(sample_mean,sample_std)

        # Probablity distribution function
        values = [value for value in range(startrange,endrange)]
        probablity = [dist.pdf(value)for value in values]
        prob = sum(probablity)
        print(f"The area between range({startrange},{endrange}):{prob:.3f}")
        return prob
    
    def stdNDGraph(dataset):
        # Converting Normal distributuion to Standard Normal distribution
        import seaborn as sns   # Used for plotting graph(Histogram)
        import numpy as np
        mean = dataset.mean()   # Calculating the Mean
        std = dataset.std()     # Calculating Standard deviaton

        values = [i for i in dataset]  # One line for loop to create the values as a list 

        z_score = [((j-mean)/std) for j in values]  # Formula for calculating Z-score 

        sns.distplot(z_score,kde=True)   # Plot the graph using Z-Score

        # Checking mean and standard deviation
        z_mean = sum(z_score)/len(z_score)
        print(f"Mean of Z- Score= {z_mean:.3f}")

        z_std = np.std(z_score)
        print(f"Standard Deviation of Z-Score= {z_std:.3f}")

        return z_mean,z_std

    def cal_vif(X):
        Vif = pd.DataFrame()  # Empty Dataframe
        Vif["variables"] = X.columns  # Create column variales in Dataframe and X.columns are the elements in the column
        Vif["VIF"] = [variance_inflation_factor(X.values,i) for i in range (X.shape[1])] 
        # for loop takes the index of each element and assigns to i then VIF is calculated and stored in VIF column
        return Vif # Returns the Vif dataFrame
