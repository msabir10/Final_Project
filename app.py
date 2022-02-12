#from msilib.schema import Binary
#from turtle import pd
from flask import Flask, render_template, redirect, url_for, request
import os
import pandas as pd
import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
import psycopg2
from config import postgres_pass, heroku_pass, heroku_URI
import analyze
import yfinance as yf

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = heroku_URI
db = SQLAlchemy(app)
    
def get_db_connection():


    h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    h_database = 'd9j5jmck4g9so5'
    h_user = 'qneoyqliwppucg'
    h_password = heroku_pass

    l_host = 'localhost'
    l_database = 'final_project'
    l_user = 'postgres'
    l_password = postgres_pass

    conn = psycopg2.connect(host=h_host, port = 5432, database=h_database, user=h_user, password=h_password)
    return conn

@app.route("/", methods =["GET", "POST"])
def index():
    
    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create initial table
    cur.execute("CREATE TABLE predicted_price (predictedprice FLOAT); INSERT INTO predicted_price (predictedprice) VALUES (1.1)")
    
    cur.execute("SELECT * from predicted_price;")
    predicted_price = cur.fetchall()
    predicted_price = list(predicted_price)[0][0]
    
    # Create initial table
    ticker = "ADBE"
    ticker_dict = yf.Ticker(ticker)
    t_info = ticker_dict.info

    # Get Ticker Features
    tick_df = pd.DataFrame(list(t_info.items()))
    tick_df = tick_df.transpose()
    new_header = tick_df.iloc[0] 
    for i in range(len(new_header)):
        new_header[i] = new_header[i].lower()
    new_header
    tick_df = tick_df[1:] 
    tick_df.columns = new_header

    #Selected Ticker to Postgres
    tick_df.to_sql(name='ticker', con=engine, if_exists='replace')
    
    cur.execute("SELECT * from ticker;")
    ticker = cur.fetchall()
    business_summary = ticker[0][4]
    current_price = ticker[0][150]
    tik = ticker[0][52]
    rec = analyze.recommendation(current_price, predicted_price)
    cur.close()
    conn.close()

    return render_template('index.html', cp=current_price, pp=predicted_price, bs=business_summary, tk=tik, rec = rec)

@app.route("/analyze", methods =["GET", "POST"])
def analyzer():

    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    analyze.run_all(ticker)
    #rec = analyze.recommendation(current_price, predicted_price)
    return redirect('/', code=302)

#cur.close()
#conn.close()
    
if __name__ == "__main__":
    app.run(debug= True)
