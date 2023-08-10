
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load():
    data= pd.read_csv("datasets/hitters.csv")
    return data



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x:'%.3f'%x)
pd.set_option('display.max_rows', None)


df=load()


df.head()
#Bağımlı değişken salary


def check_df(dataframe):
    print("########### Shape #############")
    print(dataframe.shape)
    print("########### Columns ###############")
    print(dataframe.columns)
    print("######### data type ##############")
    print(dataframe.dtypes)
    print("############## Na number #########:")
    print(dataframe.isnull().sum())
    print("######## QUANTILE ##############")
    print(dataframe.quantile([0.00, 0.05, 0.50, 0.95,0.99, 1.00]).T)

check_df(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_col= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == 'O' and dataframe[col].nunique() > car_th]
    cat_col= cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    #Num cols
    num_col= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f'Observations : {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_col, num_col, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, cat_cols, plot=False):
    print(pd.DataFrame({'Value Number': dataframe[cat_cols].value_counts(),
                        'Ratio': dataframe[cat_cols].value_counts()/len(dataframe)}))



for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)
    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True)



##### Analise to target ##########

def target_analyse(dataframe,target , col_name): #Numeric columns
    print(dataframe.groupby(target).agg({col_name : ['mean', 'count']}))

for col in num_cols:
    target_analyse(df,'Outcome', col)

def target_cat_analyse(dataframe, target, cat_col):
     print(pd.DataFrame({"TARGET_MEAN ": dataframe.groupby(cat_col)[target].mean(),
                        "Count": dataframe[cat_col].value_counts(),
                        "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe)}), end="\n\n\n")


#######KORELASYON###################33

df[num_cols].corr() #buda bi dataframe gönderir.

# Korelasyon Matrisi

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["Churn"]).sort_values(ascending=False)



### FEATURE ENGİNEERİNG

####EKSİK DEĞER ANALİZİ

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns=[col for col in df.columns if df[col].isnull().sum()>0] #Boş olan coluumnları getirir.
    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False) #sıralanmış olarak döndürür.
    ratio= (dataframe[na_columns].isnull().sum()/dataframe.shape[0] *100).sort_values(ascending=False)# yüzdesel döndürür.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True) #Boş değerleri direk silmen gerek bağımlı değişken olduğu için çünkü değiştirdiğin değer sapmaya yol açabilir.

df.dropna(inplace=True)




##################################
# AYKIRI DEĞER ANALİZİ
##########################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3= dataframe[col_name].quantile(q3)
    IQR = quantile3-quantile1
    low_limit = quantile1-1.5*IQR
    upper_limit =quantile3+1.5*IQR
    return low_limit, upper_limit


def check_outlier(dataframe, col_name):
    low_lim, up_lim= outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_lim) | (dataframe[col_name] < low_lim)].any(axis= None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    low_lim, up_lim= outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name]<low_lim), col_name]= low_lim
    dataframe.loc[(dataframe[col_name]>up_lim), col_name]= up_lim


for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


cat_cols, num_cols, cat_but_car = grab_col_names(df)


new_num_cols = [col for col in num_cols if col not in ["Salary", "Years"]]

df[new_num_cols] = df[new_num_cols] + 1


df.columns = [col.upper() for col in df.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(df)


################# Özellik Çıkarımı######################


# CAREER RUNS RATIO
df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
# CAREER BAT RATIO
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
# CAREER HITS RATIO
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
# CAREER HMRUN RATIO
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
# CAREER RBI RATIO
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
# CAREER WALKS RATIO
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["NEW_C_HIT_RATE"] = df["CHITS"] / df["CATBAT"]
# PLAYER TYPE : RUNNER
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
# PLAYER TYPE : HIT AND RUN
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN HITS
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN ALL SHOTS
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]

#Annual Averages
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]



# PLAYER LEVEL
df.loc[(df["YEARS"] <= 2), "NEW_YEARS_LEVEL"] = "Junior"
df.loc[(df["YEARS"] > 2) & (df['YEARS'] <= 5), "NEW_YEARS_LEVEL"] = "Mid"
df.loc[(df["YEARS"] > 5) & (df['YEARS'] <= 10), "NEW_YEARS_LEVEL"] = "Senior"
df.loc[(df["YEARS"] > 10), "NEW_YEARS_LEVEL"] = "Expert"



# PLAYER LEVEL X DIVISION

df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

# Player Promotion to Next League
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"




num_cols
cat_cols
cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_col = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype not in ['int64', 'float64']]

def label_encoder(dataframe, col_name):
    labelEncoder=LabelEncoder()
    dataframe[col_name] = labelEncoder.fit_transform(dataframe[col_name])
    return dataframe

for col in binary_col:
    label_encoder(df, col)

def one_hot_encoder(dataframe, col_names, dropfirst=False):
    dataframe = pd.get_dummies(dataframe, columns=col_names, drop_first=dropfirst)
    return dataframe

df=one_hot_encoder(df, cat_cols, dropfirst=True)



# 7. Robust-Scaler
for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])


# MODEL OLUŞTURMADA LOGİSTİC REGRESSİON


y = df[["SALARY"]]

X = df.drop(["SALARY"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1) #Aynı sonuçları almak için rastgeleliği bir yere sabitliyorsun random state ile


reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# b + w*x
# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

#linear regression y_hat = b + w*x
np.inner(X_train.iloc[2, :].values ,reg_model.coef_) + reg_model.intercept_
y_train.iloc[2]

np.inner(X_train.iloc[4, :].values ,reg_model.coef_) + reg_model.intercept_
y_train.iloc[4]



##########################
# Tahmin Başarısını Değerlendirme
##########################


# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))



# TRAIN RKARE
reg_model.score(X_train, y_train)



# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))


