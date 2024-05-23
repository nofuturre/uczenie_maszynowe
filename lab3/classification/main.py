import pandas as pd
from functions import univariateAnalysis_numeric1

if __name__ == "__main__":
    data = pd.read_csv('mushroom_cleaned.csv')

    df_num = data.select_dtypes(include = ['float64', 'int64'])
    lstnumericcolumns = list(df_num.columns.values)
    print(lstnumericcolumns)

    for x in lstnumericcolumns:
        univariateAnalysis_numeric1(data, x, 20)
