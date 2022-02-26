#from msilib.schema import Binary
#from turtle import pd
from flask import Flask, render_template, redirect, url_for, request
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
import psycopg2
#from config import postgres_pass#, heroku_pass, heroku_URI
import analyze
import yfinance as yf
import plotly
import plotly.express as px
from boto.s3.connection import S3Connection
import json

#Heroku secret key config
s3 = S3Connection(os.environ['heroku_pass'], os.environ['heroku_URI'],os.environ['heroku_user'],os.environ['heroku_database'])
#bbb = s3.get_bucket(s3)
h_database = os.environ['heroku_database'] #bbb.get_key('heroku_database')
h_user = os.environ['heroku_user'] #bbb.get_key('heroku_user')
h_password = os.environ['heroku_pass'] #bbb.get_key('heroku_pass')
heroku_URI = os.environ['heroku_URI']

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = heroku_URI

db = SQLAlchemy(app)
    
def get_db_connection():

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
    #analyze.initialize_table()
    return conn

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
    #stock = pd.read_sql_table('ticker', db_string)
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

@app.route("/", methods =["GET", "POST"])
def index():
    
    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    conn = get_db_connection()
    #db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    
    h_URI = heroku_URI
    db_string = h_URI

    #engine = create_engine(db_string) 
    #cur = conn.cursor()
    predicted_price_df = pd.read_sql_table('predicted_price', db_string)
    #cur.execute("SELECT * from predicted_price;")
    #predicted_price = cur.fetchall()
    predicted_price = float(predicted_price_df.predictedprice)
    predicted_accuracy = predicted_price_df.accuracy[0]

    ticker_df = pd.read_sql_table('ticker', db_string)

    #cur.execute("SELECT * from ticker;")
    #ticker = cur.fetchall()
    business_summary = ticker_df.longbusinesssummary[0]
    current_price = float(ticker_df.regularmarketprice)
    tik = ticker_df.shortname[0]
    
    sm = ticker_df.symbol[0].upper()
    w52_high = ticker_df.fiftytwoweekhigh[0]
    w52_low = ticker_df.fiftytwoweeklow[0]
    day_high = ticker_df.dayhigh[0]
    day_low = ticker_df.daylow[0]

    rec = analyze.recommendation(current_price, predicted_price)
    #cur.close()

    graphJSON = create_plot()

    conn.close()

    symbol = ticker_df.symbol[0]
    return render_template('index.html', cp=current_price, pp=predicted_price, bs=business_summary, tk=tik, rec = rec, pa=predicted_accuracy, graphJSON = graphJSON, sm = sm, w52_high = w52_high, w52_low = w52_low, day_high = day_high, day_low = day_low)

@app.route("/analyze", methods =["GET", "POST"])
def analyzer():

    if request.method == "POST":
        ticker = request.form.get("inputticker")
    
    analyze.run_all(ticker)

    return redirect('/', code=302)

    
if __name__ == "__main__":
    app.run(debug= True)
