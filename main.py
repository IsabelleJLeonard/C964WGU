import sns as sns
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import scikitplot as skplt

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

knnModel = KNeighborsRegressor()

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df.head()

df.columns = [
    "id",
    "gender",
    "age",
    'hypertension',
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose",
    "bmi",
    "smoking_status",
    "stroke", ]

# DROP ID< NOT RELEVANT.
df = df.drop("id", axis=1)

# DROP WORK TYPE-NOT RELEVANT
df = df.drop("work_type", axis=1)

# DROP SMOKING- NOT RELEVANT.
df = df.drop("smoking_status", axis=1)

df.info()

# change those to string from int64
df[['hypertension', 'heart_disease', 'stroke']] = df[['hypertension', 'heart_disease', 'stroke']].astype(str)

# double check to make sure it changed.
df.info()

# check the details to make sure no missing values
df.describe()
# there is missing values, we wil fix that later.

# distribution of numerical graph
fig, ax = plt.subplots(3, 1, figsize=(20, 15))
plt.suptitle('Distribution of Numerical', fontsize=25, color='teal')
sns.kdeplot(x=df['age'], hue=df['stroke'], shade=True, ax=ax[0], palette='ocean')
ax[0].set(xlabel='Age')
sns.kdeplot(x=df['avg_glucose'], hue=df['stroke'], shade=True, ax=ax[1], palette='twilight')
ax[1].set(xlabel='Average Glucose Level')
sns.kdeplot(x=df['bmi'], hue=df['stroke'], shade=True, ax=ax[2], palette='viridis')
ax[2].set(xlabel='Body Mass Index')

# check the glucose and bmi, it got better
df.describe()

# the bar graph.
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
ax[1, 3].axis('off')
ax[1, 2].axis('off')
ax[1, 1].axis('off')
ax[1, 0].axis('off')

plt.suptitle('Count plot for various categorical features', fontsize=30, color='teal')

ax1 = sns.countplot(x='gender', data=df, ax=ax[0, 0], palette='RdPu')
ax1.set(xlabel='Gender of the patient')

ax2 = sns.countplot(x='hypertension', data=df, ax=ax[0, 1], palette='winter')
ax2.set(xlabel='Hypertension')

ax3 = sns.countplot(x='heart_disease', data=df, ax=ax[0, 2], palette='summer')
ax3.set(xlabel='Heart disease')

ax4 = sns.countplot(x='ever_married', data=df, ax=ax[0, 3], palette='ocean')
ax4.set(xlabel='Married/ Not Married')

# the bar graph shows other in gender, we will drop that one.
df.drop(df[df['gender'] == 'Other'].index, inplace=True)

# check any null values
df.isnull().sum()


def impute(df, na_target):
    df = df.copy()

    numeric_df = df.select_dtypes(np.number)
    non_na_columns = numeric_df.loc[:, numeric_df.isna().sum() == 0].columns

    y_train = numeric_df.loc[numeric_df[na_target].isna() == False, na_target]
    X_train = numeric_df.loc[numeric_df[na_target].isna() == False, non_na_columns]
    X_test = numeric_df.loc[numeric_df[na_target].isna() == True, non_na_columns]

    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    df.loc[df[na_target].isna() == True, na_target] = y_pred

    return df


# fix the null values in bmi
df = impute(df, 'bmi')

# check
df.isnull().sum()

# check the table again.
df.head()

# drop stroke from the table and will use it to predict
X = df.drop(['stroke'], axis=1)
y = df['stroke']

# s = StandardScaler()
# df[['avg_glucose', 'age']] = s.fit_transform(df[['avg_glucose', 'age']])

df.head()

# convert everything to numerical, then drop first column.
df = pd.get_dummies(df, drop_first=True)

# check the columns
df.head()

# rename the columns, i did not like the ones with end of _1.
df.columns = ['age', 'avg_glucose', 'bmi', 'gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
              'stroke']

# check the columns to make sure they are corrected.
df.head()

oversample = RandomOverSampler(sampling_strategy='minority')
X = df.drop(['stroke'], axis=1)
y = df['stroke']
X_over, y_over = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print(prediction)

skplt.metrics.plot_confusion_matrix(y_test, prediction, figsize=(6, 6), cmap='YlGnBu');
print('Accuracy:', accuracy_score(y_test, prediction))
plt.savefig("accuracy")

# GRAPH VISUALIZATION AND DATA.
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# HEAT MAP
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, linewidth=0.5, fmt='0.2f')
plt.savefig("heat")

# DISTRIBUTION NUMERICAL
fig, ax = plt.subplots(3, 1, figsize=(20, 15))
plt.suptitle('Distribution of Numerical features based on target variable', fontsize=25, color='teal')
sns.kdeplot(x=df['age'], hue=df['stroke'], shade=True, ax=ax[0], palette='ocean')
ax[0].set(xlabel='Age')
sns.kdeplot(x=df['avg_glucose_level'], hue=df['stroke'], shade=True, ax=ax[1], palette='twilight')
ax[1].set(xlabel='Average Glucose Level')
sns.kdeplot(x=df['bmi'], hue=df['stroke'], shade=True, ax=ax[2], palette='viridis')
ax[2].set(xlabel='Body Mass Index')
plt.savefig("distrubtion numerical.png")

#  BAR GRAPH
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
ax[1, 3].axis('off')
ax[1, 2].axis('off')
ax[1, 1].axis('off')

plt.suptitle('Count plot for various categorical features', fontsize=30, color='teal')

ax1 = sns.countplot(x='gender', data=df, ax=ax[0, 0], palette='RdPu')
ax1.set(xlabel='Gender of the patient')
ax2 = sns.countplot(x='hypertension', data=df, ax=ax[0, 1], palette='winter')
ax2.set(xlabel='Hypertension')
ax3 = sns.countplot(x='heart_disease', data=df, ax=ax[0, 2], palette='summer')
ax3.set(xlabel='Heart disease')
ax4 = sns.countplot(x='ever_married', data=df, ax=ax[0, 3], palette='ocean')
ax4.set(xlabel='Married/ Not Married')
ax5 = sns.countplot(x='Residence_type', data=df, hue='stroke', ax=ax[1, 0], palette='copper')
ax5.set(xlabel='Residence Type')
plt.savefig("bar.png")


# KNN to user's input from questions.
def predict_stroke(age, avg_glucose, bmi, gender, hypertension, heart_disease, ever_married, Residence_type):
    array = [age, avg_glucose, bmi, gender, hypertension, heart_disease, ever_married, Residence_type]

    result = np.array(array, dtype=float)
    result = result.reshape(1, -1)

    solution = knn.predict(result)
    return solution


# QUESTION TAB for user to type their answers in.

def question():
    html_temp = """
            <div style="background:#025246 ;padding:10px">
            <h2 style="color:white;text-align:center;">Stroke Prediction Risk</h2>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)

    gender = st.radio('Male or Female:', ['Male', 'Female'])
    if gender == 'Male':
        gender = 1
    if gender == 'Female':
        gender = 0
    age = st.text_input('Age', '')
    bmi = st.text_input('BMI', '')
    avg_glucose = st.text_input('Glucose Level', '')
    hypertension = st.radio('Does that person have hypertension?', ['Yes', 'No'])
    if hypertension == 'Yes':
        hypertension = 1
    if hypertension == 'No':
        hypertension = 0
    heart_disease = st.radio('Does that person have heart disease?', ['Yes', 'No'])
    if heart_disease == 'Yes':
        heart_disease = 1
    if heart_disease == 'No':
        heart_disease = 0
    ever_married = st.radio("Married?", ['Yes', 'No'])
    if ever_married == 'Yes':
        ever_married = 1
    if ever_married == 'No':
        ever_married = 0
    Residence_type = st.radio("Residence type:", ['Urban', 'Rural'])
    if Residence_type == 'Urban':
        Residence_type = 1
    if Residence_type == 'Rural':
        Residence_type = 0

    if st.button("Predict the chance of getting stroke"):
        output = predict_stroke(age, avg_glucose, bmi, gender, hypertension, heart_disease, ever_married,
                                Residence_type)

        if output == [1]:
            output = 'Your risk is higher.'
        elif output == [0]:
            output = 'Your risk is low'

        st.write("If you see value error, that means one or more text-fields are empty, please go back and answer "
                 "every question, if you dont, ignore this message")

        st.success('The chance is {}'.format(output))
        st.write(
            "Here is the result, the chance to have stroke, This is based prediction if it is low or high of chance."
            "It does not mean you would not have stroke, it shows lower chance of getting one")


# GRAPH TAB to save all visualization and graph data.

def graph():
    st.title("DATA VISUALIZATION")

    # THE HEAT MAP

    st.header("The heat map")

    image = Image.open("heat.png")

    st.image(image, "The Heat Map")

    st.write("The heat map shows that the risks for people to have stroke, prediction is based on this graph from the "
             "data. 1.0 means 100 percent chance of stroke.")

    # THE DISTRIBUTION NUMBERICAL

    st.header("The distribution numerical")

    number = Image.open("distrubtion numerical.png")

    st.image(number, "The distribution numerical")

    st.write("The distribution numerical show the percentage of stroke or non stroke within its own category.")

    # THE BAR GRAPH

    st.header("The bar graph")

    bar = Image.open("bar.png")

    st.image(bar, "The bar graph")

    st.write("The bar graph show individual correlation with each factor that have a chance to increase the risk")

    # SAMPLE OF DATA COLLECTED

    st.header("Sample of Data collected")

    data = Image.open("data.jpeg")

    st.image(data, "The Data")

    st.write("The data shows 15 person out of the sample that were used into the prediction project")

    st.header("The accuracy percentage of this data")
    accuracy = Image.open("accuracy.png")
    st.image(accuracy, "Accuracy: 0.9727366255144033")

    st.write("The matrix shows the accuracy of this prediction based on KNN method.")


# FRONT WINDOW

def main():
    st.title("Welcome")
    st.header("HealthX")

    st.write("Isabelle Matthews")

    menu = ["Log In"]
    choice = st.sidebar.selectbox("menu", menu)

    if choice == "Log In":

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Log in"):
            st.sidebar.write("Uncheck the box when you need to log out")
            if username == 'user':
                if password == 'password':
                    st.sidebar.success("Logged In as {}".format(username))

                    selected_box = st.sidebar.selectbox(
                        'Choose one of the following',
                        ('Stroke Prediction', 'Graph')
                    )
                    if selected_box == 'Stroke Prediction':
                        question()
                    if selected_box == 'Graph':
                        graph()
                else:
                    st.warning("wrong password")
            else:
                st.warning("wrong username")


if __name__ == "__main__":
    main()
