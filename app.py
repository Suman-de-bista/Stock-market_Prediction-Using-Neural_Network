import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

st.title('📈 STOCK MARKET ANALYSIS AND PREDICTION USING NEURAL NETWORKS 📉')

def format_func(option):
    return CHOICES[option]

CHOICES = {1: "ADBL", 2: "CZBIL", 3: "NBB", 4: "SANIMA",5:"SBI"}
option = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
print(option)
st.write(f"You selected {format_func(option)}")
bank = format_func(option)
if st.button("ANALYZE"):
    if bank == "ADBL":
        dataset = pd.read_csv('Banks\ADBL.csv',index_col=0)
        dataset=dataset.dropna()
        dataset.set_index("Date",inplace = True)
        dataset
    elif bank == "CZBIL":
        dataset = pd.read_csv('Banks\CZBIL.csv',index_col=0)
        dataset=dataset.dropna()
        dataset.set_index("Date",inplace = True)
        dataset
        
    elif bank == "NBB":
        dataset = pd.read_csv('NBB.csv',index_col=0)
        dataset=dataset.dropna()
        dataset.set_index("Date",inplace = True)
        dataset
    elif bank == "SANIMA":
        dataset = pd.read_csv('Banks\SANIMA.csv',index_col=0)
        dataset=dataset.dropna()
        dataset.set_index("Date",inplace = True)
        dataset
    elif bank == "SBI":
        dataset = pd.read_csv('Banks\SBI.csv',index_col=0)
        dataset=dataset.dropna()
        dataset.set_index("Date",inplace = True)
        dataset


if st.button("PREDICT"):
        if bank == "ADBL":
            regressor = load_model("models\ADBL.h5")
            dataset = pd.read_csv('Banks\ADBl.csv',index_col=0)
            dataset=dataset.dropna()
            dataset_train = dataset.head(1776)
            dataset_test = dataset.tail(445)
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,506):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real ADBL Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ADBL Stock Price')
            plt.title('ADBL Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('ADBL stock Price')
            plt.legend()
           # plt.show()
            plt.savefig('static\\ADBL.png')
            image = Image.open("static\\ADBL.png")
            st.image(image, use_column_width=True)
            


        elif bank == "CZBIL":
            regressor = load_model("models\CZBIL.h5")
            dataset = pd.read_csv('Banks\CZBIL.csv')
            dataset=dataset.dropna()
            dataset_train = dataset.head(1803)
            dataset_test = dataset.tail(451)
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,512):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real CZBIL Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted CZBIL Stock Price')
            plt.title('CZBIL Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('CZBIL stock Price')
            plt.legend()
            # plt.show()
            plt.savefig('static\\CZBIL.png')
            image = Image.open("static\\CZBIL.png")
            st.image(image, use_column_width=True)


        elif bank == "NBB":
            regressor = load_model("NBB.h5")
            dataset = pd.read_csv('NBB.csv')
            dataset=dataset.dropna()
            dataset_train = dataset.head(1736)
            dataset_test = dataset.tail(434)
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,495):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real NBB Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted NBB Stock Price')
            plt.title('NBB Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('NBB stock Price')
            plt.legend()
            # plt.show()
            plt.savefig('static\\NBB.png')
            image = Image.open("static\\NBB.png")
            st.image(image, use_column_width=True)


        elif bank == "SANIMA":
            regressor = load_model("models\SANIMA.h5")
            dataset = pd.read_csv('Banks\SANIMA.csv')
            dataset=dataset.dropna()
            dataset_train = dataset.head(1633)
            dataset_test = dataset.tail(409)
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,470):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real SANIMA Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted SANIMA Stock Price')
            plt.title('SANIMA Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('SANIMA stock Price')
            plt.legend()
            # plt.show()
            plt.savefig('static\\SANIMA.png')
            image = Image.open("static\\SANIMA.png")
            st.image(image, use_column_width=True)
            
        elif bank == "SBI":
            regressor = load_model("models\SBI.h5")
            dataset = pd.read_csv('Banks\SBI.csv')
            dataset=dataset.dropna()
            dataset_train = dataset.head(1276)
            dataset_test = dataset.tail(320)
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,381):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real SBI Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted SBI Stock Price')
            plt.title('SBI Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('SBI stock Price')
            plt.legend()
            # plt.show()
            plt.savefig('static\\SBI.png')
            image = Image.open("static\\SBI.png")
            st.image(image, use_column_width=True)
        

        
    

        


