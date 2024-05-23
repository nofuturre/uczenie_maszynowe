from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def readData():
    df = pd.read_csv('mushroom_cleaned.csv')

    X = df[['cap-diameter','cap-shape','gill-attachment','gill-color','stem-height','stem-width','stem-color','season']]  # Features
    y = df['class']  # Label

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def printAccuracy(y_test,y_pred):
    # Accuracy Score on test dataset
    accuracy_test = accuracy_score(y_test,y_pred)
    print('Accuracy score on test dataset: ', accuracy_test)

    labels = [0,1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

def univariateAnalysis_numeric1(data, column,nbins):
    print("\nDescription of " + column)
    print("----------------------------------------------------------------------------")
    print(data[column].describe(),end=' ')

    print("\nCentral values of " + column)
    print("----------------------------------------------------------------------------")
    #Central values 
    print('\nMinimum : ', data[column].min(),end=' ')
    print('\nMaximum : ', data[column].max(),end=' ')
    print('\nMean value : ', data[column].mean(),end=' ')
    print('\nMedian value : ', data[column].median(),end=' ')
    print('\nStandard deviation : ', data[column].std(),end=' ')
    print('\nNull values : ', data[column].isnull().any(),end=' ')
    print('\nNull values : ', data[column].isnull().sum().sum(),end=' ')

    print("\nQuartile of " + column)
    print("----------------------------------------------------------------------------")
    #Quartiles
    Q1=data[column].quantile(q=0.25)
    Q3=data[column].quantile(q=0.75)
    print('1st Quartile (Q1) is: ', Q1)
    print('3st Quartile (Q3) is: ', Q3)
    print('Interquartile range (IQR) is ', stats.iqr(data[column]))

    print("\nOutlier detection from Interquartile range (IQR) " + column)
    print("----------------------------------------------------------------------------")
    L_outliers=Q1-1.5*(Q3-Q1)
    U_outliers=Q3+1.5*(Q3-Q1)
    print('\nLower outliers range: ', L_outliers)
    print('\nUpper outliers range: ', U_outliers)
    print('Number of outliers in upper : ', data[data[column]>U_outliers][column].count())
    print('Number of outliers in lower : ', data[data[column]<L_outliers][column].count())
    print('% of Outlier in upper: ',round(data[data[column]>U_outliers][column].count()*100/len(data)), '%')
    print('% of Outlier in lower: ',round(data[data[column]<L_outliers][column].count()*100/len(data)), '%')

    #boxplot
    plt.figure()
    print("\nBoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    ax = sns.boxplot(x=data[column])
    plt.show()
    
    #distplot
    plt.figure()
    print("\ndistplot of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(data[column])
    plt.show()
    
    #histogram
    plt.figure()
    print("\nHistogram of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(data[column], kde=False, color='red')
    plt.show()

    # Plotting mean, median and mode
    plt.figure()
    print("\nHistogram with mean, median and mode of " + column)
    print("----------------------------------------------------------------------------")
    mean=data[column].mean()
    median=data[column].median()
    mode=data[column].mode()

    print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])
    plt.hist(data[column],bins=100,color='lightblue') #Plot the histogram
    plt.axvline(mean,color='green',label='Mean')     # Draw lines on the plot for mean median and the two modes we have in GRE Score
    plt.axvline(median,color='blue',label='Median')
    plt.axvline(mode[0],color='red',label='Mode1')
    plt.legend()              # Plot the legend
    plt.show()
