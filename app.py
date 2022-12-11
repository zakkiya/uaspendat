import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.sidebar.title('Yuk Cek Obat Kolesterol Yang Sesuai Untukmu!!')
st.sidebar.write('By : Zakkiya Fitri Nur S (200411100097)')
st.title("Applikasi Web Datamining")

# inisialisasi data 
data = pd.read_csv('https://raw.githubusercontent.com/zakkiya/dataminingProject/master/drug200%20(1).csv')
tab1, tab2, tab3, tab4 = st.tabs(["Description Data", "Preprocessing Data", "Modeling", "Implementation"])

with tab1:

    st.subheader("Deskripsi Dataset")
    image = Image.open('drug.jpg')
    st.image(image)
    st.write("DRUG CLASSIFICATION")
    st.caption("""Dataset ini merupakan dataset yang menjelaskan 
    tentang klasifikasi obat kolesterol yang cocok untuk di berikan kepada pengidap penyakit kolesterol. 
    Dari dataset ini, pemberian obat yang sesuai dilihat dari umur, jenis kelamin, tingkat tekanan darah serta 
    tingkat kolesterol dari data tersebut.
    """)
    st.write("""
    ### Want to learn more?
    - Dataset [kaggel.com](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Data Understanding
    Pada dataset ini ada 5 features, yaitu :
    1. Age dengantype data numerik. Age digunakan untuk menegetahui umur pasien
    2. Sex dengan type data kategorikal. Sex digunakan untuk mengetahui jenis kelamin pasien yaitu F(Female) dan M(Male)
    3. BP dengan type data kategorikal. BP digunakan untuk menegetahui tungkat tekanan darah pasien
    4. Na_to_K dengan type data numerik. Natok digunakan untuk mengetahui asupan natrium, kalium , rasio na_k
    5. Drug dengan type data kategorikal. Drug digunakan untuk nantinya sebagai hasil klasfikasi semua fitur untuk menentukan obat yang cocok diberikan kepada pasien.
    """)

with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    df = pd.read_csv('https://raw.githubusercontent.com/zakkiya/dataminingProject/master/drug200%20(1).csv')
    st.write(data)

    proc = st.checkbox("Hasil Normalisasi")
    
    if proc:

        df['Sex'] = pd.Categorical(df["Sex"])
        df["Sex"] = df["Sex"].cat.codes
        df['BP'] = pd.Categorical(df["BP"])
        df["BP"] = df["BP"].cat.codes
        df['Cholesterol'] = pd.Categorical(df["Cholesterol"])
        df["Cholesterol"] = df["Cholesterol"].cat.codes

        data = pd.DataFrame(df)
        st.write("normalisasi feature sex,bp,cholesterol")
        st.write(data)

        # Min_Max Normalisasi
        data_drop_colum_categori = data.drop(columns = ['Age', 'Na_to_K','Drug'])
        # st.write(data_drop_colum_categori)

        data_drop_colum_drug = data.drop(columns = ['Age','Sex', 'BP','Cholesterol', 'Na_to_K'])
        # st.write(data_drop_colum_drug)
        data_drop_colum = data.drop(columns = ['Sex', 'BP','Cholesterol'])
        # st.write(data_drop_colum)
        st.write("normalisasi feature age dan na_to_k")
        from sklearn.preprocessing import MinMaxScaler
        data_for_minmax_scaler=pd.DataFrame(data_drop_colum, columns = ['Age', 'Na_to_K'])
        data_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)

        data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ['Age', 'Na_to_K'])
        st.write(data_hasil_minmax_scaler)

        data_drop_colum_na = data_hasil_minmax_scaler.drop(columns = ['Na_to_K'])
        # st.write(data_drop_colum_na)

        data_drop_colum_age = data_hasil_minmax_scaler.drop(columns = ['Age'])
        # st.write(data_drop_colum_age)
        st.write("Gabung seluruh feature yang sudah normalisasi")
        df_new = pd.concat([data_drop_colum_na,data_drop_colum_categori,data_drop_colum_age,data_drop_colum_drug], axis=1)
        st.write(df_new)

# supaya jalan
df['Sex'] = pd.Categorical(df["Sex"])
df["Sex"] = df["Sex"].cat.codes
df['BP'] = pd.Categorical(df["BP"])
df["BP"] = df["BP"].cat.codes
df['Cholesterol'] = pd.Categorical(df["Cholesterol"])
df["Cholesterol"] = df["Cholesterol"].cat.codes
data = pd.DataFrame(df)
# Min_Max Normalisasi
data_drop_colum_categori = data.drop(columns = ['Age', 'Na_to_K','Drug'])

data_drop_colum_drug = data.drop(columns = ['Age','Sex', 'BP','Cholesterol', 'Na_to_K'])

data_drop_colum = data.drop(columns = ['Sex', 'BP','Cholesterol'])

from sklearn.preprocessing import MinMaxScaler
data_for_minmax_scaler=pd.DataFrame(data_drop_colum, columns = ['Age', 'Na_to_K'])
data_for_minmax_scaler.to_numpy()
scaler = MinMaxScaler()
data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)
data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ['Age', 'Na_to_K'])

data_drop_colum_na = data_hasil_minmax_scaler.drop(columns = ['Na_to_K'])

data_drop_colum_age = data_hasil_minmax_scaler.drop(columns = ['Age'])

df_new = pd.concat([data_drop_colum_na,data_drop_colum_categori,data_drop_colum_age,data_drop_colum_drug], axis=1)
# supaya jalan
with tab3:

    X=df_new.iloc[:,0:5].values
    y=df_new.iloc[:,5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")
    model4 = st.checkbox("Ensamble Stacking")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "gaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)
    if model4:
        estimators = [
            ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('knn_1', KNeighborsClassifier(n_neighbors=10))             
            ]
        model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        filename = "stacking.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Ensamble Stacking : ",score)

with tab4:
    df['Sex'] = pd.Categorical(df["Sex"])
    df["Sex"] = df["Sex"].cat.codes
    df['BP'] = pd.Categorical(df["BP"])
    df["BP"] = df["BP"].cat.codes
    df['Cholesterol'] = pd.Categorical(df["Cholesterol"])
    df["Cholesterol"] = df["Cholesterol"].cat.codes
    data = pd.DataFrame(df)
    
    # Min_Max Normalisasi
    data_drop_colum_categori = data.drop(columns = ['Age', 'Na_to_K','Drug'])

    data_drop_colum_drug = data.drop(columns = ['Age','Sex', 'BP','Cholesterol', 'Na_to_K'])
    
    data_drop_colum = data.drop(columns = ['Sex', 'BP','Cholesterol'])
    
    from sklearn.preprocessing import MinMaxScaler
    data_for_minmax_scaler=pd.DataFrame(data_drop_colum, columns = ['Age', 'Na_to_K'])
    data_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    data_hasil_minmax_scaler=scaler.fit_transform(data_for_minmax_scaler)
    data_hasil_minmax_scaler = pd.DataFrame(data_hasil_minmax_scaler,columns = ['Age', 'Na_to_K'])
    
    data_drop_colum_na = data_hasil_minmax_scaler.drop(columns = ['Na_to_K'])
    
    data_drop_colum_age = data_hasil_minmax_scaler.drop(columns = ['Age'])

    df_new = pd.concat([data_drop_colum_na,data_drop_colum_categori,data_drop_colum_age,data_drop_colum_drug], axis=1)

    st.subheader("Parameter Inputan")
    Age = st.number_input("Masukkan Age :")
    Sex = st.radio("Masukan Jenis Kelamin (0 = Female, 1 = Male)",[0,1])
    BP = st.selectbox("Masukan Tekanan Darah (0 = HIGH, 1 = LOW, 2 = Normal)",[0,1,2])
    Cholesterol = st.selectbox("Masukan Tingkat Colesterol (0 = HIGH, 1 = NORMAL)",[0,1])
    Na_to_K = st.number_input("Masukkan Natrium to Kalium :")
    hasil = st.button("Cek klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new.iloc[:,0:5].values
    y=df_new.iloc[:,5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [Age, Sex,	BP,	Cholesterol, Na_to_K]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        st.success(f"Hasil Akurasi : {score}")