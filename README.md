# DSAI_HW2
Stock action prediction:
This is a NN model trained by IBM 5 years data. 
The data columns only include close, open, high and low of the stock price.
I transform this information to some technical indicators of trading and use them as the condition of buy, hold and sell.
After the program is successfully run, you will get the action 1(buy), 0(hold), -1(sell).

p.s. The maximum position is 1 unit of stock in our senario
Next time I'll try to use RL with data including timestamp, close, open, high, low and volume 

1. Set up virtual environment:
* pipenv or `conda creat --name <name>`
* `pipenv install pandas`

2. Make sure there are training.csv and testing.csv, and then run the code by pipenv:
* `pipenv run python app.py --training training.csv --testing testing1.csv --output output.csv`

3. Get the trading action in output.csv
