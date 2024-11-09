import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
class Diamond_Price_Predict:
    def __init__(self,data):
        try:
            self.data=data
            print(self.data.head())
            print(f"Shape of the Data set is : {self.data.shape}")
            self.data=self.data.drop("Unnamed: 0",axis=1)
            print(self.data.isnull().sum())
        except FileNotFoundError:
            print(f"Error : File Not Found")
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")

    #relationship between the carat and the price of the diamond
    def carat_price(self):
        try:
            fig=px.scatter(data_frame=self.data,
                           x='carat',
                           y='price',
                           color='cut',
                           size='depth',
                           trendline='ols',
                           title='relationship between the carat and the price of the diamond'
                           )
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    #Now I will add a new column to this dataset by calculating the size (length x width x depth) of the diamond
    def add_cols(self):
        try:
            self.data['size']=self.data['x'] * self.data['y'] * self.data['z']
            print(self.data)
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    #relationship between the size and the price of the diamond
    def size_price(self):
        try:
            fig=px.scatter(data_frame=self.data,
                           x='size',
                           y='price',
                           size='size',
                           trendline='ols',
                           color='cut',
                           title='relationship between the size and the price of the diamond')
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    #the prices of all the types of diamonds based on their colour
    def price_all_diamonds(self):
        try:
            fig=px.box(data_frame=self.data,
                       x='cut',
                       y='price',
                       color='color',
                       title='the prices of all the types of diamonds based on their colour')
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    # the prices of all the types of diamonds based on their clarity
    def diamond_prices_clarity(self):
        try:
            fig=px.box(data_frame=self.data,
                       x='cut',
                       y='price',
                       color='clarity',
                       title='the prices of all the types of diamonds based on their clarity'
                       )
            fig.show()
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    # the correlation between diamond prices and other features in the dataset
    def corr_data(self):
        try:
            correlation=self.data[['price','carat','size','x','y','z','table','depth']].corr()
            print(f"Correlation of data along with Price :\n {correlation['price'].sort_values(ascending=False)}")
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    #Train the model
    def train_model(self):
        try:
            self.data['cut']=self.data['cut'].map({"Ideal":1,
                                                   "Premium":2,
                                                   "Good":3,
                                                   "Very Good":4,
                                                   "Fair":5})
            X=np.array(self.data[['carat','cut','size']])
            y=np.array(self.data[['price']])
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=0.1,random_state=42)
            self.model=RandomForestRegressor()
            self.model.fit(self.X_train,self.y_train)
            return self.model
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
    #Giving New points data get prediction value
    def New_point_prediction(self,model):
        try:
            print("Diamond Price Prediction")
            a=float(input('Carat Size : '))
            b=int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
            c=float(input("Size : "))
            features=np.array([[a,b,c]])
            print(f"Predicted Diamond's Price =  {model.predict(features)}")
        except Exception as e:
            error_type,error_msg,err_line=sys.exc_info()
            print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")
if __name__=="__main__":
    try:
        object=Diamond_Price_Predict(data=pd.read_csv('diamonds.csv'))
        object.carat_price()
        object.add_cols()
        object.size_price()
        object.price_all_diamonds()
        object.diamond_prices_clarity()
        object.corr_data()
        model=object.train_model()
        object.New_point_prediction(model)
    except Exception as e:
        error_type, error_msg, err_line = sys.exc_info()
        print(f"Error type {error_type.__name__}-> error msg {error_msg}-> error line no is {err_line.tb_lineno}")