#from msilib.schema import Binary
#from turtle import pd
from flask import Flask, render_template, redirect, url_for, request
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
import psycopg2
from config import postgres_pass #, heroku_pass, heroku_URI
import analyze
import yfinance as yf
from boto.s3.connection import S3Connection

#Heroku secret key config
#s3 = S3Connection(os.environ['heroku_pass'], os.environ['heroku_URI'],os.environ['heroku_user'],os.environ['heroku_database'])
#bbb = s3.get_bucket(s3)
#h_database = os.environ['heroku_database'] #bbb.get_key('heroku_database')
#h_user = os.environ['heroku_user'] #bbb.get_key('heroku_user')
#h_password = os.environ['heroku_pass'] #bbb.get_key('heroku_pass')
#heroku_URI = os.environ['heroku_URI']

app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = heroku_URI
db = SQLAlchemy(app)
    
def get_db_connection():

    #h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    #h_database = heroku_database
    #h_user = heroku_user
    #h_password = heroku_pass

    l_host = 'localhost'
    l_database = 'final_project'
    l_user = 'postgres'
    l_password = postgres_pass

    conn = psycopg2.connect(host=l_host, port = 5432, database=l_database, user=l_user, password=l_password)

    db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    #h_URI = heroku_URI
    
    #db_string = h_URI
    engine = create_engine(db_string) 

    return conn

@app.route("/", methods =["GET", "POST"])
def index():
    
    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    conn = get_db_connection()
    db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    #h_URI = heroku_URI
    
    #db_string = h_URI
    engine = create_engine(db_string) 
    cur = conn.cursor()
    
    #cur.execute("SELECT * from predicted_price;")
    #predicted_price = cur.fetchall()
    
    pred_price = pd.read_sql("select * from \"predicted_price\"", conn)

    predicted_price = int(pred_price['predictedprice'])
    
    #cur.execute("SELECT * from ticker;")
    #ticker = cur.fetchall()
    
    ticker = pd.read_sql("select * from \"ticker\"", conn)

    business_summary = ticker['longbusinesssummary'][0]
    current_price = int(ticker['regularmarketprice'])
    tik = ticker['shortname'][0]
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
