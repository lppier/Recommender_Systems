import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.head()

# Data Preprocessing

df['Description'] = df['Description'].str.strip()  # remove spaces
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)  # drop all rows where invoiceNo is missing
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]  # drop all rows where invoiceNo has C in it

new_df = df
new_df.groupby(["InvoiceNo", "Description"])["Quantity"].sum()
basket = pd.pivot_table(new_df, values='Quantity', index='InvoiceNo', columns='Description').fillna(0)

basket_sets = basket.applymap(encode_units)  # if > 1 set to 1
basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.035, use_colnames=True)
frequent_itemsets.groupby(["support"])

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

frequent_itemsets.to_csv("./frequent_itemsets.csv")
rules.to_csv("./rules.csv")
