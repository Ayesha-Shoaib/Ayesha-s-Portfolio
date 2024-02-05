# Ayesha-s-Portfolio
Data Science Portfolio

# Portfolio Aanalysis
```
I focused on collecting and analyzing data for a Canadian investor's portfolio, comprising a riskless asset (T-bill ETF) and a risky asset (S&P500 stock). Historical data was gathered for the Canadian T-bill ETF and S&P500 stock index from May 19, 2000, to June 6, 2018. Additionally, CAD/USD exchange rates were utilized to convert S&P500 returns to CAD.

For the T-bill ETF, various metrics were calculated, including gross annual bond yield, log returns, deannualized bond yield, and daily percentage change. Graphs depicting the daily net bond yield and log returns were plotted, and averages and standard deviations of returns were computed.

For the S&P500, calculations included daily net return in USD and its conversion to CAD, as well as the corresponding gross net return. Graphs for daily net return and log returns were generated, along with analyses for the S&P500 ETF stock index.

I then delved into constructing different weighted portfolios. The initial steps involved computing historical averages, variances, and standard deviations for the risky asset. Subsequently, various weighted portfolios, including equal weights and dynamically re-weighted portfolios, were created and analyzed. Time series graphs were plotted, and the Sharpe Ratio was calculated for portfolio evaluation.

The fourth part aimed at constructing an optimal mean-variance portfolio for investors with varying risk aversions. Excess returns were computed for different risk aversion levels, revealing that high-risk-averse individuals tended to allocate less weight to the risky asset. Mean, variance and Sharpe Ratio were determined for each investor, and portfolio returns and values over time were visualized.

The final stage centered on testing for a unit root in the S&P500 risky asset's price and return. Augmented Dickey-Fuller (ADF) tests were employed to assess the presence of a unit root. The results indicated a unit root in the price series, leading to the adoption of an ARIMA model for compensation.

For daily returns, ADF tests were conducted to evaluate the presence of a random walk. Results suggested an absence of significant evidence for a random walk but indicated the lack of a trend or drift. Consequently, an ARMA model was chosen, emphasizing autoregressive and weighted moving average processes.

I used R and Python both.

R:

install.packages(c("tidyquant", "ggplot2", "TTR", "Metrics", "tseries", "forecast"))

library(tidyquant)
library(ggplot2)
library(TTR)
library(Metrics)
library(tseries)
library(forecast)

# Load data using tidyquant
symbols <- c("CADUSD=X", "^IRX", "^GSPC")
start_date <- "2000-05-19"
end_date <- "2018-06-06"

# Fetch data
financial_data <- tq_get(symbols, from = start_date, to = end_date)

# Data Preprocessing
financial_data <- financial_data %>%
  tq_transmute(select = adjusted,
               mutate_fun = periodReturn,
               period = "daily",
               col_rename = "return") %>%
  tk_tbl()

# Calculate log returns, deannualized bond yield, and other required metrics
financial_data <- financial_data %>%
  mutate(log_returns = log(1 + return),
         deannualized_yield = return / 253)

# Portfolio Analysis
weights <- c(0.5, 0.5)
financial_data <- financial_data %>%
  mutate(portfolio_return = weights[1] * deannualized_yield + weights[2] * return)

# Plotting Graphs
ggplot(financial_data, aes(x = date)) +
  geom_line(aes(y = deannualized_yield, color = "Daily Net Bond Yield")) +
  geom_line(aes(y = log_returns, color = "Log Returns Bond")) +
  labs(title = "Bond Yield Analysis", x = "Date", y = "Value", color = "Series") +
  theme_minimal()

# Sharpe Ratio
sharpe_ratio <- (mean(financial_data$portfolio_return) - mean(financial_data$deannualized_yield)) /
                sd(financial_data$portfolio_return)

# Unit Root Testing and ARIMA Modeling
adf_test_price <- adf.test(financial_data$adjusted, alternative = "stationary")

adf_test_return <- adf.test(financial_data$return, alternative = "stationary")

```Python:

```bash
pip install yfinance
```

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Download S&P500 data
sp500_data = yf.download('^GSPC', start='2000-05-19', end='2018-06-06')

# Download Canadian 3-month treasury bills data (I'm using T-bill ETF as an example)
bond_data = yf.download('BIL', start='2000-05-19', end='2018-06-06')

# Choose another source for exchange rate data, or skip this step if not needed

# Combine datasets
combined_data = pd.merge(bond_data, sp500_data, left_index=True, right_index=True, how='inner')

# Handle missing data
combined_data = combined_data.ffill().bfill()

# Data Preprocessing
combined_data['log_returns_bond'] = np.log(1 + combined_data['Close_x'].pct_change())
combined_data['deannualized_yield'] = combined_data['Close_x'] / 253

# Portfolio Analysis
weights = {'bond': 0.5, 'sp500': 0.5}
combined_data['portfolio_return'] = weights['bond'] * combined_data['deannualized_yield'] + \
                                     weights['sp500'] * combined_data['Close_y'].pct_change()

# Calculate mean, variance, standard deviation
mean_portfolio = combined_data['portfolio_return'].mean()
variance_portfolio = combined_data['portfolio_return'].var()
std_dev_portfolio = combined_data['portfolio_return'].std()

# Plotting Graphs
plt.plot(combined_data.index, combined_data['deannualized_yield'], label='Daily Net Bond Yield')
plt.plot(combined_data.index, combined_data['log_returns_bond'], label='Log Returns Bond')
plt.title('Bond Yield Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Sharpe Ratio
sharpe_ratio = (mean_portfolio - combined_data['deannualized_yield'].mean()) / std_dev_portfolio

```
Unit Root Testing and ARIMA Modeling
result_price = sm.tsa.adfuller(combined_data['Close_y'])
Check the result and choose the appropriate model for price


result_return = sm.tsa.adfuller(combined_data['portfolio_return'].dropna())
```

# Speech Recognition

   ```bash
   pip install Flask SpeechRecognition
   ```

   ```python
   from flask import Flask, render_template, request
   import speech_recognition as sr

   app = Flask(__name__)

   @app.route("/", methods=["GET", "POST"])
   def index():
       if request.method == "POST":
           # Check if the post request has the file part
           if "file" not in request.files:
               return render_template("index.html", error="No file part")

           file = request.files["file"]

           # If the user does not select a file, browser also
           # submit an empty part without filename
           if file.filename == "":
               return render_template("index.html", error="No selected file")

           if file:
               try:
                   # Perform Speech Recognition
                   recognizer = sr.Recognizer()
                   audio = sr.AudioFile(file)
                   with audio as source:
                       audio_data = recognizer.record(source)

                   # Recognize speech using Google Speech Recognition
                   text = recognizer.recognize_google(audio_data)

                   return render_template("index.html", text=text)

               except sr.UnknownValueError:
                   return render_template("index.html", error="Speech Recognition could not understand audio")

               except sr.RequestError as e:
                   return render_template("index.html", error=f"Could not request results from Google Speech Recognition service; {e}")

       return render_template("index.html", error=None, text=None)

   if __name__ == "__main__":
       app.run(debug=True)
   ```

 **HTML Template**

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>Speech Recognition Platform</title>
   </head>
   <body>
       <h1>Speech Recognition Platform</h1>
       {% if error %}
           <p style="color: red;">{{ error }}</p>
       {% endif %}
       <form method="post" enctype="multipart/form-data">
           <label for="file">Upload Audio File:</label>
           <input type="file" name="file" accept=".wav, .mp3">
           <button type="submit">Submit</button>
       </form>
       {% if text %}
           <h2>Transcription:</h2>
           <p>{{ text }}</p>
       {% endif %}
   </body>
   </html>
   ```

   ```bash
   python app.py
   ```

   Go to http://127.0.0.1:5000/. 

# Handwriting Recognition:

   ```bash
   pip install tensorflow
   ```

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from tensorflow.keras.datasets import mnist
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.utils import to_categorical

   # MNIST dataset
   (X_train, y_train), (X_test, y_test) = mnist.load_data()

   
   X_train_flat = X_train.reshape((X_train.shape[0], -1)).astype('float32')
   X_test_flat = X_test.reshape((X_test.shape[0], -1)).astype('float32')

   X_train_flat /= 255.0
   X_test_flat /= 255.0

   y_train_onehot = to_categorical(y_train)
   y_test_onehot = to_categorical(y_test)

   X_train, X_val, y_train, y_val = train_test_split(X_train_flat, y_train_onehot, test_size=0.2, random_state=42)

   model = Sequential()
   model.add(Dense(128, input_dim=784, activation='relu'))
   model.add(Dense(10, activation='softmax'))

   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

   model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

   loss, accuracy = model.evaluate(X_test_flat, y_test_onehot)
   print(f'Test Accuracy: {accuracy * 100:.2f}%')

   predictions = model.predict(X_test_flat)
   fig, axes = plt.subplots(2, 5, figsize=(10, 4))
   for i, ax in enumerate(axes.flatten()):
       ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
       ax.set_title(f"True: {y_test[i]}, Predicted: {np.argmax(predictions[i])}")
   plt.show()
   ```

   ```bash
   python handwriting_recognition_tf.py
   ```

   This script will load the MNIST dataset, build and train a simple neural network, and evaluate its performance on the test set. 


   # Sentiment Analysis
```
Sure, let's recreate a simple sentiment analysis script using Python with NLTK (Natural Language Toolkit), BeautifulSoup for web scraping, and Matplotlib for visualization. Please note that web scraping should be done responsibly and in accordance with the website's terms of service.

1. **Install Required Libraries:**
   Make sure you have NLTK, BeautifulSoup, and Matplotlib installed.

   ```bash
   pip install nltk beautifulsoup4 matplotlib
   ```

2. **Create Python Script:**

   ```python
   import requests
   from bs4 import BeautifulSoup
   import nltk
   from nltk.sentiment import SentimentIntensityAnalyzer
   import pandas as pd
   import matplotlib.pyplot as plt

   # NLTK resources
   nltk.download('vader_lexicon')

   def get_news_sentiment(url):
       # Fetch HTML content
       response = requests.get(url)
       soup = BeautifulSoup(response.text, 'html.parser')

       # Extract text from HTML
       paragraphs = soup.find_all('p')
       text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

       # Analyze sentiment
       sia = SentimentIntensityAnalyzer()
       sentiment_score = sia.polarity_scores(text)['compound']
       return sentiment_score

   # URLs for stock news articles
   urls = {
       'Amazon': 'https://finance.yahoo.com/quote/AMZN/news?p=AMZN',
       'AMD': 'https://finance.yahoo.com/quote/AMD/news?p=AMD',
       'Facebook': 'https://finance.yahoo.com/quote/FB/news?p=FB'
   }

   # sentiment scores for each stock
   sentiments = {ticker: get_news_sentiment(url) for ticker, url in urls.items()}

   df = pd.DataFrame(list(sentiments.items()), columns=['Stock', 'Sentiment Score'])
   df.set_index('Stock', inplace=True)

   # Plot sentiment scores
   df.plot(kind='bar', rot=0, color='skyblue', legend=False)
   plt.title('Sentiment Analysis of Stock News')
   plt.xlabel('Stocks')
   plt.ylabel('Sentiment Score')
   plt.show()
   ```

   ```bash
   python sentiment_analysis.py
   ```

   This script fetches stock news articles for Amazon, AMD, and Facebook from Yahoo Finance, performs sentiment analysis using NLTK, and visualizes the sentiment scores with Matplotlib.


  # Smoke Data Analysis
```

library(tidyverse)
library(plm)

data <- read.csv('Smoke_data.csv')

crime_data <- as.data.frame(data)

# Impute missing values by taking the average of the previous and following year
crime_data <- crime_data %>%
  arrange(province, year) %>%
  mutate_if(is.numeric, function(x) ifelse(is.na(x), (lag(x) + lead(x)) / 2, x))

# Panel regression model
# Taking 'CR' is the dependent variable and 'UN', 'INC', 'PP', 'LI', 'YTH', 'HS' as the independent variables
model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "random")

# Test for fixed or random effects using Hausman test
hausman_test <- phtest(model, alternative = "two.sided")

# Choose the appropriate model 
if (hausman_test$p.value < 0.05) {
  # Fixed effect model is appropriate
  final_model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "within")
} else {
  # Random effect model is appropriate
  final_model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "random")
}

summary(final_model)
```

This script performs a panel regression analysis using both fixed and random effects models and selects the appropriate model based on the Hausman test. The results and coefficients are displayed in the summary.


# 
