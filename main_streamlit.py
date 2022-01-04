# This is a sample Python script
import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Data
api_url = "https://interfacescoring.bernardtoutain.repl.co"

df = pd.read_csv('test_df.csv')
df.drop(['Unnamed: 0', 'TARGET'], axis=1, inplace=True)

info_client = pd.read_csv('info_client.csv')
info_client.drop(['Unnamed: 0'], axis=1, inplace=True)

df_DAYS_BIRTH = info_client[['DAYS_BIRTH', 'PROBABILITY_payment']].copy()
df_DAYS_BIRTH = df_DAYS_BIRTH.groupby(['DAYS_BIRTH']).mean()
df_DAYS_BIRTH = df_DAYS_BIRTH.reset_index()

sns.set_style('darkgrid')

st.set_page_config(layout="wide")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Analyse profile of your customer files')

row0_2.subheader(
    'A Streamlit web app by Bernard TOUTAIN')

# api_url = "https://interfacescoring.bernardtoutain.repl.co"

# df = pd.read_csv('test_df.csv')
# df.drop(['Unnamed: 0', 'TARGET'], axis=1, inplace=True)

# info_client = pd.read_csv('info_client.csv')
# info_client.drop(['Unnamed: 0'], axis=1, inplace=True)

# df_DAYS_BIRTH = info_client[['DAYS_BIRTH', 'PROBABILITY_payment']].copy()
# df_DAYS_BIRTH = df_DAYS_BIRTH.groupby(['DAYS_BIRTH']).mean()
# df_DAYS_BIRTH = df_DAYS_BIRTH.reset_index()

st.title('Credit application')

identifiant = st.sidebar.number_input('Enter customer file number', min_value=100001, max_value=456250, step=1)

need_help = st.sidebar.expander('Need help? 👉')
with need_help:
    st.markdown(
        "Customer file number examples: 100001; 100005 ... 456221; 456222; 456223")

#predict_btn = st.sidebar.button('Predict')

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((.1, 3, .2, 1, .5))

# if predict_btn:

df_sk_id_curr = df['SK_ID_CURR'] == identifiant
filtered_df = df[df_sk_id_curr]

reponse_proba_seuil = requests.post(api_url + '/predict_proba_seuil',
                                    data=filtered_df.to_json(),
                                    headers={'Content-Type': 'application/json'})

row1_1.subheader('Prediction is:')

if reponse_proba_seuil.text == '[False]':

    row1_2.subheader('LOAN ACCEPTED')

else:
    row1_2.subheader('LOAN DENIED')

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns((.1, 3, .2, 1, .5))

reponse_proba = requests.post(api_url + '/predict_proba',
                              data=filtered_df.to_json(),
                              headers={'Content-Type': 'application/json'})

df = pd.DataFrame(json.loads(reponse_proba.text))
df.rename(columns={"0": "LOAN ACCEPTED", "1": "LOAN DENIED"}, inplace=True)
row2_1.subheader('Probability LOAN ACCEPTED is (%):')
row2_2.subheader(df['LOAN ACCEPTED'][0] * 100)
df1 = df['LOAN ACCEPTED']

row3_spacer1, row3_1, row3_spacer2, row3_2, row3_spacer3 = st.columns((.1, 3, .2, 1, .5))

row3_1.subheader('Probability LOAN DENIED is (%):')
row3_2.subheader(df['LOAN DENIED'][0] * 100)
df2 = df['LOAN DENIED']

row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3 = st.columns((.1, 2, .2, 2, .1))
with row4_1:
    fig = plt.figure()

    ax = df1.plot.bar(stacked=True, legend=False, figsize=(10, 8))

    plt.xlabel('PROBABILITY LOAN ACCEPTED', fontsize=25)
    plt.yticks(fontsize=25)
    plt.axhline(y=1 - 0.316033, color='green', linestyle='-')

    st.pyplot(fig)

with row4_2:
    fig = plt.figure()

    ax = df2.plot.bar(stacked=True, color='orange', legend=False, figsize=(10, 8))

    plt.xlabel('PROBABILITY LOAN DENIED', fontsize=25)
    plt.yticks(fontsize=25)
    plt.axhline(y=0.316033, color='r', linestyle='-')

    st.pyplot(fig)

line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

with line1_1:
    st.header('Analyzing the customer')

row5_spacer1, row5_1, row5_spacer2 = st.columns((.1, 3.2, .1))

# with row5_1:

proba_accepted = df['LOAN ACCEPTED'][0]
proba_non_accepted = 1 - (df['LOAN ACCEPTED'][0])

df_sk_id_curr = info_client['SK_ID_CURR'] == identifiant
filtered_df_2 = info_client[df_sk_id_curr]
analyzing_the_customer = filtered_df_2.drop(['PROBABILITY_payment', 'PROBABILITY_payment_default'], axis=1)
st.dataframe(data=analyzing_the_customer)

st.subheader("Customer Age")

fig, ax = plt.subplots(figsize=(10, 3))
# ax = fig.subplots()
sns.barplot(x=df_DAYS_BIRTH['DAYS_BIRTH'], y=df_DAYS_BIRTH['PROBABILITY_payment'], color='goldenrod', ax=ax)
plt.axhline(y=proba_accepted, color='green', linestyle='-')
# ax.set_xlabel('Age')
# ax.set_ylabel('Probability')
plt.xticks(fontsize=7)
plt.yticks(fontsize=10)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Probability', fontsize=15)
st.pyplot(fig)

st.subheader("Customer work")

df_DAYS_EMPLOYED = info_client[['DAYS_EMPLOYED', 'PROBABILITY_payment']].copy()
df_DAYS_EMPLOYED = df_DAYS_EMPLOYED.groupby(['DAYS_EMPLOYED']).mean()
df_DAYS_EMPLOYED = df_DAYS_EMPLOYED.reset_index()

fig, ax = plt.subplots(figsize=(10, 3))
sns.barplot(x=df_DAYS_EMPLOYED['DAYS_EMPLOYED'], y=df_DAYS_EMPLOYED['PROBABILITY_payment'], color='orangered', ax=ax)
plt.axhline(y=proba_accepted, color='green', linestyle='-')
# ax.set_xlabel('Year')
# ax.set_ylabel('Probability')
plt.xticks(fontsize=7)
plt.yticks(fontsize=10)
plt.xlabel('Years employed', fontsize=15)
plt.ylabel('Probability', fontsize=15)
st.pyplot(fig)

st.subheader("AMT_INCOME_TOTAL")

df_AMT_INCOME_TOTAL_Bin = info_client.groupby(['AMT_INCOME_TOTAL_Bin']).mean()
df_AMT_INCOME_TOTAL_Bin = df_AMT_INCOME_TOTAL_Bin.reset_index()
labels = ["until 30", "30-60", "60_90", "90_120", "120_150", "150_180", "180_210", "210_300", "300_500", "over 500"]
df_AMT_INCOME_TOTAL_Bin = df_AMT_INCOME_TOTAL_Bin.reindex(
    df_AMT_INCOME_TOTAL_Bin["AMT_INCOME_TOTAL_Bin"].map(dict(zip(labels, range(len(labels))))).sort_values().index)

fig, ax = plt.subplots(figsize=(10, 3))
sns.barplot(x=df_AMT_INCOME_TOTAL_Bin['AMT_INCOME_TOTAL_Bin'],
            y=df_AMT_INCOME_TOTAL_Bin['PROBABILITY_payment'], color='grey')
plt.axhline(y=proba_accepted, color='green', linestyle='-')
ax.set_xlabel('AMT_INCOME_TOTAL')
ax.set_ylabel('Probability')
plt.title('Year Built', size=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('AMT_INCOME_TOTAL', fontsize=15)
plt.ylabel('PROBABILITY', fontsize=15)
st.pyplot(fig)

st.subheader("AMT_CREDIT")

df_AMT_CREDIT_Bin = info_client.groupby(['AMT_CREDIT_Bin']).mean()
df_AMT_CREDIT_Bin = df_AMT_CREDIT_Bin.reset_index()

labels = ["until 60", "60_90", "90_120", "120_150", "150_180", "180_210", "210_300", "300_500", "over 500"]
df_AMT_CREDIT_Bin = df_AMT_CREDIT_Bin.reindex(
    df_AMT_CREDIT_Bin["AMT_CREDIT_Bin"].map(dict(zip(labels, range(len(labels))))).sort_values().index)

fig, ax = plt.subplots(figsize=(10, 3))
sns.barplot(x=df_AMT_CREDIT_Bin['AMT_CREDIT_Bin'], y=df_AMT_CREDIT_Bin['PROBABILITY_payment'],
            color="cadetblue")
plt.axhline(y=proba_accepted, color='green', linestyle='-')
ax.set_xlabel('AMT_CREDIT')
ax.set_ylabel('Probability')
plt.title('Year Built', size=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('AMT_CREDIT', fontsize=15)
plt.ylabel('PROBABILITY', fontsize=15)
st.pyplot(fig)

st.subheader("AMT_ANNUITY")

df_AMT_ANNUITY_Bin = info_client.groupby(['AMT_ANNUITY_Bin']).mean()
df_AMT_ANNUITY_Bin = df_AMT_ANNUITY_Bin.reset_index()

labels = ["until 10", "10_30", "30_50", "50_70", "70_90", "90_110", "110_130", "130_150", "150_170", "over 170"]
df_AMT_ANNUITY_Bin = df_AMT_ANNUITY_Bin.reindex(
    df_AMT_ANNUITY_Bin["AMT_ANNUITY_Bin"].map(dict(zip(labels, range(len(labels))))).sort_values().index)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=df_AMT_ANNUITY_Bin['AMT_ANNUITY_Bin'], y=df_AMT_ANNUITY_Bin['PROBABILITY_payment'],
            color='darkseagreen')
plt.axhline(y=proba_accepted, color='green', linestyle='-')
ax.set_xlabel('AMT_ANNUITY')
ax.set_ylabel('Probability')
plt.title('Year Built', size=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('AMT_ANNUITY', fontsize=15)
plt.ylabel('PROBABILITY', fontsize=15)
st.pyplot(fig)

st.subheader("Comparison between client file ans group of clients")



target = [0, 1]
var1 = st.sidebar.selectbox('Choose TARGET', target, help='Filter report to show only one hospital')
st.write(f'Target:{var1}')

DAYS_BIRTH = ["until 25", "25_30", "30_35", "35_40", "40_45", "45_50", "50_55", "55_60", "60_65", "over 65"]
var2 = st.sidebar.selectbox('Choose DAYS_BIRTH', DAYS_BIRTH, help='Filter report to show only one hospital')
st.write(f'DAYS_BIRTH:{var2}')

DAYS_EMPLOYED = ["until 5", "5_10", "10_15", "15_20", "20_25", "25_30", "30_35", "35_40", "40_45", "over 45"]
var3 = st.sidebar.selectbox('Choose DAYS_EMPLOYED', DAYS_EMPLOYED, help='Filter report to show only one hospital')
st.write(f'DAYS_EMPLOYED:{var3}')

AMT_INCOME_TOTAL = ["until 30", "30_60", "60_90", "90_120", "120_150", "150_180", "180_210", "210_300", "300_500",
                    "over 500"]
var4 = st.sidebar.selectbox('Choose AMT_INCOME_TOTAL', AMT_INCOME_TOTAL, help='Filter report to show only one hospital')
st.write(f'AMT_INCOME_TOTAL:{var4}')

AMT_CREDIT = ["until 60", "60_90", "90_120", "120_150", "150_180", "180_210", "210_300", "300_500", "over 500"]
var5 = st.sidebar.selectbox('Choose AMT_CREDIT', AMT_CREDIT, help='Filter report to show only one hospital')
st.write(f'AMT_CREDIT:{var5}')

AMT_ANNUITY = ["until 10", "10_30", "30_50", "50_70", "70_90", "90_110", "110_130", "130_150", "150_170", "over 170"]
var6 = st.sidebar.selectbox('Choose AMT_ANNUITY', AMT_ANNUITY, help='Filter report to show only one hospital')
st.write(f'AMT_ANNUITY:{var6}')

#var1 = 0
#var2 = '50_55'
#var3 = '5_10'
#var4 = '120_150'
#var5 = 'over 500'
#var6 = '10_30'

info_client_f = info_client[(info_client['TARGET'] == var1) & (info_client['DAYS_BIRTH_Bin'] == var2) & \
                            (info_client['DAYS_EMPLOYED_Bin'] == var3) & (info_client['AMT_INCOME_TOTAL_Bin'] == var4) & \
                            (info_client['AMT_CREDIT_Bin'] == var5) & (info_client['AMT_ANNUITY_Bin'] == var6)]



#st.dataframe(info_client)
st.dataframe(info_client_f)

st.subheader("Customer Age")
# age boxplot
Data = info_client_f[['DAYS_BIRTH']].copy()
DAYS_BIRTH = analyzing_the_customer['DAYS_BIRTH'].iloc[0]

row6_spacer1, row6_1, row6_spacer2, row6_2, row6_spacer3 = st.columns((.1, 2, .2, 2, .1))

with row6_1:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=Data)
    plt.axhline(y=DAYS_BIRTH, color='r', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('DAYS_BIRTH', fontsize=25)
    st.pyplot(fig)

with row6_2:
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax = fig.subplots()
    sns.barplot(x=info_client_f['DAYS_BIRTH'], y=info_client_f['PROBABILITY_payment'], color='goldenrod', ax=ax)
    plt.axhline(y=proba_accepted, color='green', linestyle='-')
    # ax.set_xlabel('Age')
    # ax.set_ylabel('Probability')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Age', fontsize=25)
    plt.ylabel('Probability', fontsize=25)
    st.pyplot(fig)

st.subheader("Customer work")

row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.1, 2, .2, 2, .1))


with row7_1:

    Data = info_client_f[['DAYS_EMPLOYED']].copy()
    DAYS_EMPLOYED = analyzing_the_customer['DAYS_EMPLOYED'].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=Data, showfliers=False)
    plt.axhline(y=DAYS_EMPLOYED, color='r', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('DAYS_EMPLOYED', fontsize=25)
    st.pyplot(fig)

with row7_2:
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax = fig.subplots()
    sns.barplot(x=info_client_f['DAYS_EMPLOYED'], y=info_client_f['PROBABILITY_payment'], color='goldenrod', ax=ax)
    plt.axhline(y=proba_accepted, color='green', linestyle='-')
    # ax.set_xlabel('Age')
    # ax.set_ylabel('Probability')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Customer work', fontsize=25)
    plt.ylabel('Probability', fontsize=25)
    st.pyplot(fig)

st.subheader("AMT_INCOME_TOTAL")

row8_spacer1, row8_1, row8_spacer2, row8_2, row8_spacer3 = st.columns((.1, 2, .2, 2, .1))


with row8_1:

    Data = info_client_f[['AMT_INCOME_TOTAL']].copy()
    AMT_INCOME_TOTAL = analyzing_the_customer['AMT_INCOME_TOTAL'].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=Data, showfliers=False)
    plt.axhline(y=AMT_INCOME_TOTAL, color='r', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_INCOME_TOTAL', fontsize=25)
    st.pyplot(fig)

with row8_2:

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(x=info_client_f['AMT_INCOME_TOTAL'], y=info_client_f['PROBABILITY_payment'], color='goldenrod', ax=ax)
    plt.axhline(y=proba_accepted, color='green', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_INCOME_TOTAL', fontsize=25)
    plt.ylabel('Probability', fontsize=25)
    st.pyplot(fig)

st.subheader("AMT_CREDIT")

row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3 = st.columns((.1, 2, .2, 2, .1))

with row9_1:

    Data = info_client_f[['AMT_CREDIT']].copy()
    AMT_CREDIT = filtered_df['AMT_CREDIT'].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=Data, showfliers=False)
    plt.axhline(y=AMT_CREDIT, color='r', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_CREDIT', fontsize=25)
    st.pyplot(fig)


with row9_2:

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=info_client_f['AMT_CREDIT'], y=info_client_f['PROBABILITY_payment'], color='goldenrod', ax=ax)
    plt.axhline(y=proba_accepted, color='green', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_CREDIT', fontsize=25)
    plt.ylabel('Probability', fontsize=25)
    st.pyplot(fig)

st.subheader("AMT_ANNUITY")

row10_spacer1, row10_1, row10_spacer2, row10_2, row10_spacer3 = st.columns((.1, 2, .2, 2, .1))

with row10_1:

    Data = info_client_f[['AMT_ANNUITY']].copy()
    AMT_ANNUITY = filtered_df['AMT_ANNUITY'].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=Data, showfliers=False)
    plt.axhline(y=AMT_ANNUITY, color='r', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_ANNUITY', fontsize=25)
    st.pyplot(fig)

with row10_2:


    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=info_client_f['AMT_ANNUITY'], y=info_client_f['PROBABILITY_payment'], color='goldenrod', ax=ax)
    plt.axhline(y=proba_accepted, color='green', linestyle='-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('AMT_ANNUITY', fontsize=25)
    plt.ylabel('Probability', fontsize=25)
    st.pyplot(fig)


