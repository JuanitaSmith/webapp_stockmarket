# Stock Market Analysis Dashboard - PythonAnywhere Deployment

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Usage](#usage)
4. [Website](#webapp)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [References](#ref)

## 1. Installation <a name="installation"></a>
The `requirements.txt` file lists all Python libraries that are required for this project and they can be installed using:
```pip install -r requirements.txt```

## 2. Project Motivation<a name="motivation"></a>

This project demonstrate basic web development skills to build skills for other types of data science applications. 
Data scientists are increasingly being asked to deploy their work as an application in the cloud.

The objective of this project was to learn how to:
- Use front end languages HTML, CSS, and Javascript to design and add functionality to webpages
- Use Bootstrap to make front end development easier
- Use Plotly to create charts for great visualizations
- Program in Flask to provide the backend to the website
- Package files for deployment

The project extract Goodyear 'GT' stock market data using API from [marketstack](https://marketstack.com), for which a subscription is needed.
end point: 'https://api.marketstack.com/v1/eod'

## 3. Usage<a name="usage"></a>
To run it locally:
1. Go to stockmarket.py and uncomment line `app.run(host='0.0.0.0', port=3001, debug=True)`
2. run `stockmarket.py` in your shell. Go to http://localhost:3001 to view the site.

The project will use the data already extracted from the API, stored in directory `data`.

If you want to extract latest data from the API:
1) Subscribe minimum to marketstack.com basic plan
2) Get API key from marketstack.com
3) Store API key locally in environment variable 'MARKET_STACK_API' or adjust paramter name in wrangling_scripts/wrangle_data in gather_data
4) Delete the `data` folder to trigger the refresh

A log file `marketstack.log` will be created on execution

## 4. Website<a name="webapp"></a>

Webapp stock analysis results can be seen [here](http://juasmithy.pythonanywhere.com)
user name: js
password: dev

## 5. Acknowledgements<a name="licensing"></a>

* [Marketstack](https://marketstack.com)
* [Udacity](https://www.udacity.com/)


## 6. References<a name="ref"></a>

- [How to setup virtual environment for your project](https://help.pythonanywhere.com/pages/Virtualenvs/)
- [How to setup webapp in PythonAnywhere](https://pub.towardsai.net/its-time-to-say-goodbye-to-heroku-and-welcome-pythonanywhere-ec3a2b8caa3b)