import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from itertools import combinations
import gc # for garbage collection

# Correct base path for the dataset
file_path= '/home/akash/Desktop/MasterThesis/NF-CSE-CIC-IDS2018-cleaned.csv'


# Load the CSV file
df = pd.read_csv(file_path)




df.fillna(0, inplace=True)

def data_cleaning(df):
    df.columns=df.columns.str.strip()
    print("Dataset Shape: ",df.shape)
    
    num=df._get_numeric_data()
    num[num<0]=0
    
    zero_variance_cols=[]
    for col in df.columns:
        if len(df[col].unique()) == 1:
            zero_variance_cols.append(col)
    df.drop(columns=zero_variance_cols,axis=1,inplace=True)
    print("Zero Variance Columns: ",zero_variance_cols, " are dropped!!")
    print("Shape after removing the zero variance columns: ",df.shape)
    
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    print(df.isna().any(axis=1).sum(), "rows dropped")
    df.dropna(inplace=True)
    print("Shape after Removing NaN: ",df.shape)
    
    df.drop_duplicates(inplace=True)
    print("Shape after dropping duplicates: ",df.shape)
    
    column_pairs = [(i,j) for i,j in combinations(df,2) if df[i].equals(df[j])]
    ide_cols=[]
    for col_pair in column_pairs:
        ide_cols.append(col_pair[1])
    df.drop(columns=ide_cols,axis=1,inplace=True)
    print("Columns which have identical values: ",column_pairs," dropped!")
    print("Shape after removing identical value columns: ",df.shape)
    df.columns=df.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','')
    columns_to_remove = ['ipv4_src_addr', 'ipv4_dst_addr', 'attack']
    df = df.drop(columns=columns_to_remove)
    return df
df=data_cleaning(df)


print(df.info())

print(df.label.value_counts())


size=len(df.loc[df.label==1])
print(size)
bal_df=df.groupby('label').apply(lambda x: x.sample(n=min(size,len(x))))


print(bal_df.shape)


gc.collect()


print(type(bal_df.label))


# Save the processed DataFrame to a CSV file
output_path = '/home/akash/Desktop/MasterThesis/NF-CSE-CIC-IDS2018.csv'
bal_df.to_csv(output_path, index=False)
print(f"Dataset saved successfully at {output_path}")