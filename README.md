# Final_Project

Presentation Link: https://docs.google.com/presentation/d/1DhafRC4UR8gYSsYWFzmQp2ccarPR_o2Tjhe-K0d2gU8/edit?usp=sharing

## Collaborators 

The team (**Lydia Alexander**, **Muhammad Sabir**, and **Lev Levine**) has made the decision to collaborate on this project working on and submitting each area as a team.

## Communications Protocol

### Cadence & Tools

The team meets at least 3 times per week: Tuesday, Thursday, and Saturday using the **Discord** Collaboration Software. 

**Github** is being actively used to share the work and manage versions. 

**Slack** is used throught the week to exchange messages.

### Team Core Areas of Responsibility

**Lydia** - Database

**Muhammad** - Github & Vusualization

**Lev** - Machine Learning & App Architecture

# Overview/Outline

### Selected topic

  * Stock Price Prediction

### Reasons the topic was selected

Statistics:
- $95 trillion is the global stock market value.
- The current stock market crash interrupted a 10-year record.
- On average, the stock market performs the poorest in September.
- The US represents about 55.9% of the total global stock market capitalization.
- 55% of US adults have money in the stock market.
- About 10% of US households hold international equity.

[Source: Stock Market Statistics](https://spendmenot.com/blog/stock-market-statistics/#:~:text=Let%E2%80%99s%20see%20how%20many%20people%20invest%20in%20the,of%20Americans%20own%20stocks%2C%20the%20answer%20is%2052%25)

Top Reasons For Choosing:

  * The stock market is a data rich topic with data available for analysis to include APIs, libraries and static data to pull from. Having a large amount of data for our model will increase the liklihood of our model being successful.
  * This topic is very relevant to the maturity of the American people.  As the above statistics detail, 55% of US adults have money in the stock market, making this topic very significant to a large number of people in the US.
  * A member of our team is very involved in the day trading of the stock market and understands the business.

### Description of data sources

#### Yahoo Finance dataset on Kaggle  [https://www.kaggle.com/jerryhans/key-statistics-yahoo-finance-stocks-of-2003-2013](https://www.kaggle.com/jerryhans/key-statistics-yahoo-finance-stocks-of-2003-2013)

  The Yahoo Finance dataset from Kaggle includes key statistics of the S & P 500 ("a market-capitalization-weighted index of the 500 largest publicly traded companies in the U.S.") from 2003 - 2013. [Source: Investopedia] (https://www.investopedia.com/terms/s/sp500.asp)

#### Yahoo Finance API [https://www.yahoofinanceapi.com/](https://www.yahoofinanceapi.com/)

  "Yahoo Finance API is a service that provides stock quotes and other financial information for any stock ticker symbol. It is one of the largest source of financial data available on the web, and it is updated continuously throughout the day. Yahoo Finance API covers over 37,000 stocks in more than 50 countries. Yahoo Finance is a media property that provides various financial news and data including stock details, quotes, press releases, financial reports."  [Source: Yahoo Finance API - The Complete Guide](https://blog.api.rakuten.net/api-tutorial-yahoo-finance/#:~:text=Yahoo%20Finance%20API%20is%20a%20reliable%20source%20of,Finance%20API%20can%20be%20accessed%20from%20Rakuten%20RapidAPI.)

#### yfinance Library [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)

  The yfinance Libarary data source "is a popular open source library developed as a means to access the financail data available on Yahoo Finance" offering a "range of market data on stocks, bonds, currencies and crytpocurrencies." [yfinance Libarary - A Complete Guide](https://algotrading101.com/learn/yfinance-guide/#:~:text=What%20is%20yfinance%3F%20yfinance%20is%20a%20popular%20open,market%20data%20on%20stocks%2C%20bonds%2C%20currencies%20and%20cryptocurrencies.)


### Questions the App will Answer with the Data

  * Based on the data input into the NN model, provide a data-driven buy/sell recommendation to the end user on a selected stock on their interest.

### Description of Data Exploration

We were looking for the datasets that had numerous metrics and ratios around stock fundamentals that would enable training the model and predicting the stock price - the target variable.

The Kaggle data set contained more than 30 fields and a large number of data points sufficent to fit the neural net or machine learning model. We also extracted data from the yfinance library and official Yahoo Finance API. Both contained a rich set of data points for stock fundamentals as well as historical data.

Extracts from these data sets have been loaded into DataFrames in Jupyter Notebook, reviewed for missing and Null values, data types, descriptive statistics for each column, and any data anomalies. 

At the conclusion of the data exploration, we established that the Kaggle data set does meet the requirements as the main data set for the project machine learning / deep learning training.

The yfinance library does meet the requirements for the on-demand request of the current stock fundamentals.

### Description of the Analysis

We have reviewed the following potential features and analysed their impact on the predicting the stock price. 

- **DE Ratio**: The debt-to-equity (D/E) ratio compares a company's total liabilities to its shareholder equity 
- **Trailing P/E**: The price-to-earnings ratio (P/E ratio) is the ratio for valuing a company that measures its current share price relative to its earnings per share (EPS)
- **Price/Sales**: The price-to-sales ratio equals a company's market capitalization divided it by the company's total sales or revenue over the past 12 months
- **Price/Book**: The price-to-book ratio compares a company's market value to its book value                  
- **Profit Margin**: The profit margin is a ratio of a company's profit (sales minus all expenses) divided by its revenue              
- **Operating Margin**: The operating margin is the ratio of operating income to net sales            
- **Return on Assets**: The term return on assets (ROA) is a financial ratio that indicates how profitable a company is in relation to its total assets            
- **Return on Equity**: The Return on equity (ROE) is a measure of financial performance calculated by dividing net income by shareholders' equity            
- **Revenue Per Share**: The Revenue per share is a ratio that computes the total revenue earned per share over a designated period            
- **Forward P/E**: The price-earnings ratio is the ratio of a company's share price to the company's earnings per share                 
- **PEG Ratio**: The PEG ratio is a company's Price/Earnings ratio divided by its earnings growth rate                  
- **Enterprise Value/Revenue**: The Enterprise Value to Revenue Multiple equals the corporate value (equity plus debt minus cash) divided by its annual revenue    
- **Enterprise Value/EBITDA**:The Enterprise Value to EBITDA is the corporate value divided by its earnings before interest, taxes, and depreciation  
- **Gross Profit**: The Gross profit is the profit a business makes after subtracting the cost of goods sold
- **Diluted EPS**: The Earnings per share is a company's net profit divided by the number of common shares it has outstanding                 
- **Earnings Growth**: The Earnings growth is the change in an entity's reported net income over a period of time             
- **Revenue Growth**: The Revenue growth is the change in an entity's reported sales over a period of time             
- **Total Cash Per Share**: The Cash per share is the measure of available cash to a business divided by the number of equity shares outstanding       
- **Current Ratio**: The Current Ratio is Current Assets divided by Current Liabilities               
- **Book Value Per Share**: The Book value per share is the ratio of equity available to common shareholders divided by the number of outstanding shares       
- **Cash Flow**: The Cash flow is the net amount of cash and cash equivalents being transferred into and out of a business                   
- **Beta**: The Beta is a measure of a stock's volatility in relation to the overall market.                        
- **Market Cap**
- **Enterprise Value**
- **EBITDA**
- **Net Income Avl to Common Shareholders**
- **Total Cash**
- **Total Debt**

## Database

A databased was created using PostgreSQL.

## Machine Learning Model

### Description of preliminary data preprocessing

- Dropped columns that did not contribute to the prediction
- Replaced NaN (null) values with mean values for each stock ticker
- Removed remaining rows that contained NaN values 
- Removed rows with outliers beyong 3 standard deviation from the mean
- Scaled the training data to Mean = 0 and STD = 1 (with Standard Scaler)

### Description of preliminiary feature engineering and preliminary feature selection, including our decision-making process
NEED

### Description of how data was split into training and testing sets
NEED

### Explanation of model choice, including limitations and benefits
NEED




### Model Overview 

A provisional Deep Learning Regression Model has been developed (see **LOCAL_SERVER/ML_Model.ipynb**)

Target Variable: **Stock Price** (Price)

The model is using the **3 hidden layers**:

- hidden_nodes_layer1 = **80 neurons**
- hidden_nodes_layer2 = **30 neurons**
- hidden_nodes_layer3 = **10 neurons**

The **SELU** activation function is used on all the **Hidden layers**. 

The **RELU** activation function is used on all the **Output layer**. 

Since this is a regression analysis, RELU and SELU activation functions generate adequate results.

**The Neural Network Model:**

![Model](Resources/model_screen.png)

The model is run **100 epochs** to train.


### Data Transformation

- An unsupervised machine learning model has been applied to the cleaned dataset to cluster the data into groups with similar behavioral patterns. K-Means model has been used and the Elbow Curve built.

![Elbow Curve](Resources/Elbow.png)

Based on the Elbow Curve, **k=6** has been selected. We have assigned a **Cluster Class** to each Stock Ticker in the dataset accordingly.

### Model Training and Saving

For each Cluster, a unique Deep Learning Model has been trained based on the Cluster data subset. The Models for each Cluster have been saved (total of 6). The Scalers for each Cluster have been saved (total of 6).

For new stock tickers that are not part of the overall dataset and may be potentially entered by the App users, a 7th model has been trained. The 7th model is based on the overall dateset (without clustering). Both, the model and scaler have also been saved.

### Model Evaluation

Adjusted R-Square has been used to evaluate the each Model accuracy. 

The R-Square results are below.

1. Model 0: 0.34
2. Model 1: 0.23
3. Model 2: 0.71
4. Model 3: 0.99
5. Model 4: 0.37
6. Model 5: 0.62
7. Model 6: 0.33

### Overall User Experience

#### A. Explore individual stock ticker (Core Product Feature)

1. Enter stock ticker
2. The backend will pull the most recent feature data and target variable data from Yahoo Finance
3. The DL Model will calculate the predicted stock price base on the most current feature data
4. The predicted and current stock price are displayed for the user
5. If predicted stock price > current stock price, a BUY recommendation generated and vise versa

**The working Prototype 2.0 User Interface (localhost deployment):**

![App](Resources/screen2.png)

### Prototype 1.0 App Deployment on HEROKU (to be updated)

[https://stockoptimizer.herokuapp.com/](https://stockoptimizer.herokuapp.com/)

## Dashboard

Dashboard Presentation: https://docs.google.com/presentation/d/1hXU28unDzPH3O0gM9gG7vCKDZXwtV4GZLEZ8WskptY0/edit?usp=sharing

### Storyboard on Google Slides

### Description of the tool(s) that are used to create final dashboard

1. HTML
2. CSS
3. Java Script
4. Flask
5. Python Scripts

### Description of interactive element(s)
