#from msilib.schema import Binary
#from turtle import pd
from flask import Flask, render_template, redirect, url_for, request
import os
import sqlalchemy
#from flask_sqlalchemy import SQLAlchemy
import psycopg2
#from config import postgres_pass
import analyze

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://zrzqqcjhzcyaje:9208481eb51aa2fad9071ab3f78435d4e471df3a3be08a8b44e66055751b73c1@ec2-18-215-8-186.compute-1.amazonaws.com:5432/d24lp1nuba41a9'
db = sqlalchemy(app)
    
def get_db_connection():


    h_host = 'ec2-18-235-114-62.compute-1.amazonaws.com'
    h_database = 'd9j5jmck4g9so5'
    h_user = 'qneoyqliwppucg'
    h_password = '5529db771b78b297270b333ee6f96f37be1d329e1fc5d9d0891f621c69ae0f14'

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

    cur.execute("SELECT * from predicted_price;")
    predicted_price = cur.fetchall()
    predicted_price = list(predicted_price)[0][0]

  
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
