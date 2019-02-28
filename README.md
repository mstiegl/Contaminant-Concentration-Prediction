# Contaminant-Concentration-Prediction
Based on weather data and historical air quality data, this project predicts the concentration of pollutants (including PM2.5, PM10 and O3) at 35 meteorological stations in Beijing for the next 48 hours. Firstly, I use the time and space features, build the model to fill the missing value on the original data set, and then use the weather features, historical pollutant concentration values, and time features to construct the feature engineering, and then obtain more than 100 features. Finally, I built GradientBoost, RandomForest, and LightGBM. I use ensemble learning to predict the concentration of pollutants, and the smape error is about 0.5.

[For more details, you can read this report.](https://github.com/YimiaoSun/Contaminant-Concentration-Prediction/blob/master/readme.pdf)

[The data for this project](https://github.com/YimiaoSun/Contaminant-Concentration-Prediction/tree/master/dataset)

[Coding Solutions](https://github.com/YimiaoSun/Contaminant-Concentration-Prediction/tree/master/code)

Note: This project is made by SUN Yimiao and LIU Ziwan.
