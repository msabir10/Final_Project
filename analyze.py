import pandas as pd
import numpy as np
import json
import os
import requests
#from config import api_key, postgres_pass#, heroku_pass, heroku_URI
import yfinance as yf
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
import keras

from boto.s3.connection import S3Connection

#Heroku secret key config
s3 = S3Connection(os.environ['heroku_pass'], os.environ['heroku_URI'],os.environ['heroku_user'],os.environ['heroku_database'])
#bbb = s3.get_bucket(s3)
h_database = os.environ['heroku_database'] #bbb.get_key('heroku_database')
h_user = os.environ['heroku_user'] #bbb.get_key('heroku_user')
h_password = os.environ['heroku_pass'] #bbb.get_key('heroku_pass')
heroku_URI = os.environ['heroku_URI']
h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'

def initialize_table():

    h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    #h_database = heroku_database
    #h_user = heroku_user
    #h_password = heroku_pass

    #l_host = 'localhost'
    #l_database = 'final_project'
    #l_user = 'postgres'
    #l_password = postgres_pass

    conn = psycopg2.connect(host=h_host, port = 5432, database=h_database, user=h_user, password=h_password)

    #db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    h_URI = heroku_URI
    
    db_string = h_URI
    engine = create_engine(db_string) 

    #Select Ticker
    ticker = 'adbe'

    # pull Ticker details from Yahoo Finance
    ticker_dict = yf.Ticker(ticker)

    #test if Ticker is real, if not -> display ADBE
    if len(ticker_dict.info.keys())<5:
        ticker_dict = yf.Ticker('adbe')
    
    t_info = ticker_dict.info

    # Get Ticker Features and lower case for column names
    tick_df = pd.DataFrame(list(t_info.items()))
    tick_df = tick_df.transpose()
    new_header = tick_df.iloc[0] 

    for i in range(len(new_header)):
        new_header[i] = new_header[i].lower()
    #new_header
    tick_df = tick_df[1:] 
    tick_df.columns = new_header

    tick_df['symbol'] = tick_df['symbol'].str.lower()
    tick_df.symbol

    #Selected Ticker to Postgres
    tick_df.to_sql(name='tickerr', con=engine, if_exists='replace')

    # Create a DF for the Symbol Name Table (CSV)
    #class_df=pd.read_csv('ticker_clusters.csv', index_col=False, names=['tick', 'class'], header=0)
    
    # Create a new Table for Clusters in Posgres
    class_df.to_sql(name='clusters', con=engine, if_exists='replace', index=False)

    # Create initial table
    cur = conn.cursor()
    cur.execute("select * from information_schema.tables where table_name=%s", ('mytable',))
    
    #check if the table exists
    if not bool(cur.rowcount):
        ddf = {'predictedprice': 1.1, 'accuracy': 'High'}
        price_df=pd.DataFrame(ddf, index=[0])
        price_df.to_sql(name='predicted_price', con=engine, if_exists='replace')
        
    #cur.execute("""CREATE TABLE IF NOT EXISTS predicted_price (predictedprice FLOAT, accuracy VARCHAR);""")
    #cur.execute("""INSERT INTO predicted_price (predictedprice, accuracy) VALUES (1.1, 'High');""")
    #conn.commit()

    # Create a DF for the Symbol Name Table (CSV)
    class_df=pd.read_csv('ticker_clusters.csv', index_col=False, names=['tick', 'class'], header=0)
    # Create a new Table for Clusters in Posgres
    class_df.to_sql(name='clusters', con=engine, if_exists='replace', index=False)

    # Create ticker Table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ticker;")
    cur.execute("SELECT t.*, c.class INTO ticker FROM tickerr t LEFT JOIN clusters c ON t.symbol = c.tick;") 
    conn.commit()

    return None

def data_predict(ticker):

    #Connect to PostgreSQL
    
    h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    #h_database = heroku_database
    #h_user = heroku_user
    #h_password = heroku_pass

    #l_host = 'localhost'
    #l_database = 'final_project'
    #l_user = 'postgres'
    #l_password = postgres_pass

    conn = psycopg2.connect(host=h_host, port = 5432, database=h_database, user=h_user, password=h_password)

    #db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    h_URI = heroku_URI
    
    db_string = h_URI
    engine = create_engine(db_string) 

    # pull Ticker details from Yahoo Finance
    ticker_dict = yf.Ticker(ticker)

    #test if Ticker is real, if not -> display ADBE
    if len(ticker_dict.info.keys())<5:
        ticker_dict = yf.Ticker('adbe')
        
    t_info = ticker_dict.info

    # Get Ticker Features and lower case for column names
    tick_df = pd.DataFrame(list(t_info.items()))
    tick_df = tick_df.transpose()
    new_header = tick_df.iloc[0] 

    for i in range(len(new_header)):
        new_header[i] = new_header[i].lower()
    #new_header
    tick_df = tick_df[1:] 
    tick_df.columns = new_header

    tick_df['symbol'] = tick_df['symbol'].str.lower()
    tick_df.symbol

    #Selected Ticker to Postgres
    tick_df.to_sql(name='tickerr', con=engine, if_exists='replace')

    #Query the Feature data for Clusters Table
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ticker;")
    cur.execute("SELECT t.*, c.class INTO ticker FROM tickerr t LEFT JOIN clusters c ON t.symbol = c.tick;") 
    conn.commit()

    #Query the Feature data for DL Model
    cur.execute("""SELECT debttoequity, trailingpe, pricetosalestrailing12months, 
    pricetobook, profitmargins, operatingmargins, returnonassets, returnonequity, 
    revenuepershare, forwardpe, pegratio, enterprisetorevenue, enterprisetoebitda, 
    grossprofits, forwardeps, earningsgrowth, revenuegrowth, totalcashpershare, 
    currentratio, bookvalue, operatingcashflow, beta, class
    FROM ticker;""")      

    query_results = list(np.asarray(cur.fetchall()[0]))

    # Remove Nulls from Yahoo Finance Data
    for i in range(len(query_results)-1):
        if query_results[i] == None:
            query_results[i] =0
        if i == len(query_results)-1:
            if query_results[i+1] == None:
                query_results[i+1] =6 # If the stock does not belong to any cluster
        

    # Load the Saved Model
    from tensorflow import keras
    try:
        model_num = int(query_results[-1]) # Get the Model Class Number
    except:
        #if the stock is not in clusters
        model_num = 6

    query_results = query_results[:-1]

    nn = keras.models.load_model(f"Model/Stock_Optimization_Class_{model_num}.h5")

    #Accuracy Calc
    if model_num == 0:
        model_accuracy = 'Medium'
    elif model_num == 1:
        model_accuracy = 'Medium'
    elif model_num == 2:
        model_accuracy = 'High'
    elif model_num == 3:
        model_accuracy = 'Very High'
    elif model_num == 4:
        model_accuracy = 'Medium'
    elif model_num == 5:
        model_accuracy = 'Medium'
    else:
        model_accuracy = 'Medium'

    # Run the model to predict Stock Price
    query_results = np.array(query_results).reshape(-1,1)
    query_results = query_results.transpose()
    query_results.shape

    # Load the Scaler
    X_scaler = joblib.load(f'Model/Stock_Scaler_Class_{model_num}.gz')
    query_scaled = X_scaler.transform(query_results) 

    predicted_price = pd.DataFrame(data=nn.predict(query_scaled), columns=['predictedprice'], index=None)
    predicted_price['accuracy'] = model_accuracy

    #Save Predicted Price to Postgres
    #Selected Ticker to Postgres
    predicted_price.to_sql(name='predicted_price', con=engine, if_exists='replace', index = None)

    # Close the cursor and connection to so the server can allocate
    # bandwidth to other requests
    cur.close()
    conn.close()
    return None

def create_plot():

    #Connect to PostgreSQL
    
    h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    #h_database = heroku_database
    #h_user = heroku_user
    #h_password = heroku_pass

    #l_host = 'localhost'
    #l_database = 'final_project'
    #l_user = 'postgres'
    #l_password = postgres_pass

    conn = psycopg2.connect(host=h_host, port = 5432, database=h_database, user=h_user, password=h_password)

    #db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    h_URI = heroku_URI
    
    db_string = h_URI

    ######## Chart in Python ############
    stock = pd.read_sql_table('ticker', db_string)
    stock_ticker = stock.symbol[0]
    st = yf.Ticker(stock_ticker)
    period ='1y'
    interval = '1d'
    df = st.history(period=period, interval=interval)
    #hist_df

    # Graph formatting
    df=df.reset_index()
    df.columns = ['Date-Time']+list(df.columns[1:])
    max = (df['Open'].max())
    min = (df['Open'].min())
    range = max - min
    margin = range * 0.05
    max = max + margin
    min = min - margin
    fig = px.area(df, x='Date-Time', y="Open",
        hover_data=("Open","Close","Volume"), 
        range_y=(min,max), template="seaborn" )

    #fig.show()
    # Create a JSON representation of the graph
    #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #graph = graphJSON
    return graphJSON

def recommendation(cp, pp):
    if pp/cp>1.5:
        rec = "Strong Buy"
    elif pp/cp>1.1:
        rec = "Buy"
    elif pp/cp>0.9:
        rec = "Hold"
    else:
        rec = "Sell"
    return rec

def run_all(ticker):

    initialize_table()
    data_predict(ticker)
    #create_plot()
    return None

if __name__ == "__main__":
    run()

