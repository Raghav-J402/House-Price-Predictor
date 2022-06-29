<img width="960" alt="Screenshot 2022-06-29 184338" src="https://user-images.githubusercontent.com/87217567/176445070-dff98b7b-e592-4c99-96d0-11e00e724d01.png">

# House-Price-Predictor
This repostitory contains a data science project which can predict the home prices in the various locations in Banglore city using the property price dataset of banglore available in kaggle.

Next step was to write a python flask server that uses the saved model to serve http requests.

Third component is the website built in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price.

During model building it covered almost all data science concepts such as data load and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc. Technology and tools wise this project covers,

Python
Numpy and Pandas for data cleaning
Matplotlib for data visualization
Sklearn for model building
Jupyter notebook, visual studio code and pycharm as IDE
Python flask for http server
HTML/CSS/Javascript for UI

# Steps to run the website locally
1. Download the `BangloreHousePrices` folder in your `C:\\` drive

2. If you dont have pycharm, download the latest version on your device

3. Run the python file `server.py` 

4. Once the server is running go to the `clients` folder and click on `index.html` file

5. On this webpage, you can get the house price prediction based on the parameters such as location, room numbers and number of bathrooms.
  
