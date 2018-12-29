# AIR POLLUTION FORECASTING USING LSTM
---

#### Steps followed: 

1. Data preparation 
2. Data visualization
3. LSTM data preparation
4. Fit model along with regularization term
5. Evaluate model

**Data preparation:**
- Replace NA values 
- Parse date-time into pandas dataframe index
- Specified clear names for each columns

**Data visualization**
- Used matplotlib to plot

![Air pollution dataset](https://raw.githubusercontent.com/sagarmk/Air-pollution-forecasting-with-RNN/master/images/img1.png)


**LSTM data preparation**
- Normalized data 
- Transformed dataset into supervised learning problem

**Model Fitting**
- Split data into train and test 
- Split into i/p and o/p
- Reshape into 3D
- Define a 50 neuron followed by 1 nueron LSTM 
- Add dropout at 30%
- Plot history of training and testing loss

![Train and Test loss](https://raw.githubusercontent.com/sagarmk/Air-pollution-forecasting-with-RNN/master/images/img2.png)

**Evaluate model**
- Make prediction 
- Invert scalings
- Calculate RMSE




