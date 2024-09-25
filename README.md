[![Watch the video](https://img.youtube.com/vi/x9syF_Bl6O8/maxresdefault.jpg)](https://www.youtube.com/watch?v=x9syF_Bl6O8)

Click on the image above to watch a demo on YouTube!

# Overview: 
* Developed a machine learning tool using scikit-learn’s ridge regression to provide hyper-local weather predictions, accurately forecasting next-day temperatures within a 2-degree margin.
* Trained the model on historical weather data from local weather station datasets, using predictors like temp_max, temp_min, and the previous day’s precipitation to generate precise predictions.

# Problem: 
* Often, local weather stations cannot report accurate forecasts relative to your specific location.

* However, many weather stations have public repositories of historical weather statistics, which may be closer to your specific location.

# Solution: 
* Utilizing publicly available weather datasets (ie National Oceanic and Atmospheric Administration) to provide accurate machine learning location-based weather forecasts.

# Future Works:
* Ideally, I'd like to be able to predict accurately more than one day in advance while maintaining a 2-degree margin of error.

* Auto update max_temp, min_temp and precip values using APIs
