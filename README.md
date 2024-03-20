# Stock Market Analysis Dashboard - PythonAnywhere Deployment

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Usage](#usage)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The `requirements.txt` file lists all Python libraries that are required for this project and they can be installed using:
```pip install -r requirements.txt```

## Project Motivation<a name="motivation"></a>
The project extract Goodyear 'GT' stock market data using API from [marketstack](https://marketstack.com), for which a subscription is needed.
end point: 'https://api.marketstack.com/v1/eod'

## Usage<a name="usage"></a>
To run it locally:
1. Go to stockmarket.py and uncomment line `app.run(host='0.0.0.0', port=3001, debug=True)`
2. run `stockmarket.py` in your shell. Go to http://localhost:3001 to view the site.

## Acknowledgements<a name="licensing"></a>

* [Marketstack](https://marketstack.com)
* [Udacity](https://www.udacity.com/)





## How to host webapp in PythonAnywhere

[How to setup virtual environment for your project](https://help.pythonanywhere.com/pages/Virtualenvs/)
[How to setup webapp in PythonAnywher](https://pub.towardsai.net/its-time-to-say-goodbye-to-heroku-and-welcome-pythonanywhere-ec3a2b8caa3b)