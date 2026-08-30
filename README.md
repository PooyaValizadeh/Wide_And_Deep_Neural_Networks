# Wide and Deep Neural Network
### by Pooya Valizadeh

what are Wide and Deep Neural networks???

Wide & Deep Neural Networks are a type of neural network architecture introduced by Google in 2016 that combines two different types of learning into one model:

Wide part → Learns memorization (remembers specific patterns/examples)

Deep part → Learns generalization (understands new, unseen patterns)


```python
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")
```


```python
from sklearn.datasets import fetch_california_housing as data
data()
```




    {'data': array([[   8.3252    ,   41.        ,    6.98412698, ...,    2.55555556,
               37.88      , -122.23      ],
            [   8.3014    ,   21.        ,    6.23813708, ...,    2.10984183,
               37.86      , -122.22      ],
            [   7.2574    ,   52.        ,    8.28813559, ...,    2.80225989,
               37.85      , -122.24      ],
            ...,
            [   1.7       ,   17.        ,    5.20554273, ...,    2.3256351 ,
               39.43      , -121.22      ],
            [   1.8672    ,   18.        ,    5.32951289, ...,    2.12320917,
               39.43      , -121.32      ],
            [   2.3886    ,   16.        ,    5.25471698, ...,    2.61698113,
               39.37      , -121.24      ]], shape=(20640, 8)),
     'target': array([4.526, 3.585, 3.521, ..., 0.923, 0.847, 0.894], shape=(20640,)),
     'frame': None,
     'target_names': ['MedHouseVal'],
     'feature_names': ['MedInc',
      'HouseAge',
      'AveRooms',
      'AveBedrms',
      'Population',
      'AveOccup',
      'Latitude',
      'Longitude'],
     'DESCR': '.. _california_housing_dataset:\n\nCalifornia Housing dataset\n--------------------------\n\n**Data Set Characteristics:**\n\n:Number of Instances: 20640\n\n:Number of Attributes: 8 numeric, predictive attributes and the target\n\n:Attribute Information:\n    - MedInc        median income in block group\n    - HouseAge      median house age in block group\n    - AveRooms      average number of rooms per household\n    - AveBedrms     average number of bedrooms per household\n    - Population    block group population\n    - AveOccup      average number of household members\n    - Latitude      block group latitude\n    - Longitude     block group longitude\n\n:Missing Attribute Values: None\n\nThis dataset was obtained from the StatLib:\nhttps://lib.stat.cmu.edu/datasets/houses.zip\n\nThe target variable is the median house value for California districts,\nexpressed in hundreds of thousands of dollars ($100,000).\n\nThis dataset was derived from the 1990 U.S. census, using one row per census\nblock group. A block group is the smallest geographical unit for which the U.S.\nCensus Bureau publishes sample data (a block group typically has a population\nof 600 to 3,000 people).\n\nA household is a group of people residing within a home. Since the average\nnumber of rooms and bedrooms in this dataset are provided per household, these\ncolumns may take surprisingly large values for block groups with few households\nand many empty houses, such as vacation resorts.\n\nIt can be downloaded/loaded using the\n:func:`sklearn.datasets.fetch_california_housing` function.\n\n.. rubric:: References\n\n- Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,\n  Statistics and Probability Letters, 33:291-297, 1997.\n'}



## preprocessing


```python
df = pd.DataFrame(data().data, columns=data().feature_names)
df['target'] = data().target

print(df.head())
df.boxplot()
```

       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \
    0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   
    1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   
    2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   
    3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   
    4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   
    
       Longitude  target  
    0    -122.23   4.526  
    1    -122.22   3.585  
    2    -122.24   3.521  
    3    -122.25   3.413  
    4    -122.25   3.422  





    <Axes: >




    
![png](wide_and_deep_nn_files/wide_and_deep_nn_5_2.png)
    



```python

df = df[(df < 12000).all(axis=1)]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.boxplot()
```




    <Axes: >




    
![png](wide_and_deep_nn_files/wide_and_deep_nn_7_1.png)
    



```python
x = df.drop(columns="target").values
y = df["target"].values
```


```python
xtest
```




    array([[-0.43551895,  0.5800371 , -0.53872421, ..., -0.11186918,
            -1.3431042 ,  1.20649039],
           [-1.27796197,  0.50057891, -0.46524822, ..., -0.09830272,
            -0.72945816,  0.95680229],
           [-0.57864096, -0.69129397, -0.60041114, ..., -0.03593376,
            -0.66856199,  0.56728887],
           ...,
           [ 0.77261003, -1.00912674, -0.3496614 , ...,  0.12615078,
             0.79294614, -1.13558393],
           [-0.14980209,  1.85136817,  0.08628711, ..., -0.01340557,
             1.0084249 , -1.45518469],
           [-0.95144348,  0.02382976, -0.72042457, ...,  0.05093566,
            -0.74351112,  0.62721401]], shape=(4128, 8))




```python
from sklearn.model_selection import train_test_split as tts 
xtrain , xtest , ytrain , ytest = tts(x,y,random_state=26,test_size=0.2)
```


```python
print(xtrain.shape , ytrain.shape , xtest.shape , ytest.shape)
```

    (16503, 8) (16503,) (4126, 8) (4126,)



```python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)
```

## bulding the model


```python
from tensorflow.keras.layers import concatenate , Input , Dense
```


```python
x.shape
```




    (20629, 8)




```python
input_ = Input(shape = (8,))
h1 = Dense(60,activation="relu")(input_)
h2 = Dense(10,activation="relu")(h1)
h3 = concatenate([input_,h2])
output = Dense(1)(h3)
model = keras.Model(input_,output)
```


```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_6"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_7       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">540</span> │ input_layer_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">610</span> │ dense_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_6       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_layer_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │                   │            │ dense_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">19</span> │ concatenate_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,169</span> (4.57 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,169</span> (4.57 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
model.compile(loss= keras.losses.MeanSquaredError(),optimizer= "Adam",metrics=[keras.metrics.MeanSquaredError()])
```

## training the model


```python
history = model.fit(xtrain,ytrain,epochs=50,batch_size=100,validation_split=0.2)
```

    Epoch 1/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 4ms/step - loss: 1.8292 - mean_squared_error: 1.8292 - val_loss: 0.7248 - val_mean_squared_error: 0.7248
    Epoch 2/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.6362 - mean_squared_error: 0.6362 - val_loss: 0.5777 - val_mean_squared_error: 0.5777
    Epoch 3/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.5246 - mean_squared_error: 0.5246 - val_loss: 0.4793 - val_mean_squared_error: 0.4793
    Epoch 4/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.4639 - mean_squared_error: 0.4639 - val_loss: 0.4393 - val_mean_squared_error: 0.4393
    Epoch 5/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.4262 - mean_squared_error: 0.4262 - val_loss: 0.4091 - val_mean_squared_error: 0.4091
    Epoch 6/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.4146 - mean_squared_error: 0.4146 - val_loss: 0.3909 - val_mean_squared_error: 0.3909
    Epoch 7/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3916 - mean_squared_error: 0.3916 - val_loss: 0.3739 - val_mean_squared_error: 0.3739
    Epoch 8/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3847 - mean_squared_error: 0.3847 - val_loss: 0.4178 - val_mean_squared_error: 0.4178
    Epoch 9/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3760 - mean_squared_error: 0.3760 - val_loss: 0.3695 - val_mean_squared_error: 0.3695
    Epoch 10/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3674 - mean_squared_error: 0.3674 - val_loss: 0.3627 - val_mean_squared_error: 0.3627
    Epoch 11/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3606 - mean_squared_error: 0.3606 - val_loss: 0.3763 - val_mean_squared_error: 0.3763
    Epoch 12/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - loss: 0.3549 - mean_squared_error: 0.3549 - val_loss: 0.3879 - val_mean_squared_error: 0.3879
    Epoch 13/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - loss: 0.3512 - mean_squared_error: 0.3512 - val_loss: 0.3442 - val_mean_squared_error: 0.3442
    Epoch 14/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - loss: 0.3508 - mean_squared_error: 0.3508 - val_loss: 0.3341 - val_mean_squared_error: 0.3341
    Epoch 15/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 0.3412 - mean_squared_error: 0.3412 - val_loss: 0.3248 - val_mean_squared_error: 0.3248
    Epoch 16/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.3381 - mean_squared_error: 0.3381 - val_loss: 0.3203 - val_mean_squared_error: 0.3203
    Epoch 17/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3343 - mean_squared_error: 0.3343 - val_loss: 0.3158 - val_mean_squared_error: 0.3158
    Epoch 18/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3313 - mean_squared_error: 0.3313 - val_loss: 0.3138 - val_mean_squared_error: 0.3138
    Epoch 19/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3259 - mean_squared_error: 0.3259 - val_loss: 0.3095 - val_mean_squared_error: 0.3095
    Epoch 20/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3234 - mean_squared_error: 0.3234 - val_loss: 0.3042 - val_mean_squared_error: 0.3042
    Epoch 21/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3226 - mean_squared_error: 0.3226 - val_loss: 0.3403 - val_mean_squared_error: 0.3403
    Epoch 22/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3261 - mean_squared_error: 0.3261 - val_loss: 0.3065 - val_mean_squared_error: 0.3065
    Epoch 23/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3156 - mean_squared_error: 0.3156 - val_loss: 0.3116 - val_mean_squared_error: 0.3116
    Epoch 24/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3190 - mean_squared_error: 0.3190 - val_loss: 0.3137 - val_mean_squared_error: 0.3137
    Epoch 25/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3116 - mean_squared_error: 0.3116 - val_loss: 0.3034 - val_mean_squared_error: 0.3034
    Epoch 26/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3121 - mean_squared_error: 0.3121 - val_loss: 0.3085 - val_mean_squared_error: 0.3085
    Epoch 27/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3087 - mean_squared_error: 0.3087 - val_loss: 0.3040 - val_mean_squared_error: 0.3040
    Epoch 28/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3070 - mean_squared_error: 0.3070 - val_loss: 0.2994 - val_mean_squared_error: 0.2994
    Epoch 29/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3045 - mean_squared_error: 0.3045 - val_loss: 0.3114 - val_mean_squared_error: 0.3114
    Epoch 30/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3052 - mean_squared_error: 0.3052 - val_loss: 0.3011 - val_mean_squared_error: 0.3011
    Epoch 31/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.3039 - mean_squared_error: 0.3039 - val_loss: 0.2920 - val_mean_squared_error: 0.2920
    Epoch 32/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3007 - mean_squared_error: 0.3007 - val_loss: 0.3383 - val_mean_squared_error: 0.3383
    Epoch 33/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3122 - mean_squared_error: 0.3122 - val_loss: 0.3022 - val_mean_squared_error: 0.3022
    Epoch 34/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3004 - mean_squared_error: 0.3004 - val_loss: 0.3115 - val_mean_squared_error: 0.3115
    Epoch 35/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2963 - mean_squared_error: 0.2963 - val_loss: 0.3130 - val_mean_squared_error: 0.3130
    Epoch 36/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2950 - mean_squared_error: 0.2950 - val_loss: 0.3222 - val_mean_squared_error: 0.3222
    Epoch 37/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.3063 - mean_squared_error: 0.3063 - val_loss: 0.2956 - val_mean_squared_error: 0.2956
    Epoch 38/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2983 - mean_squared_error: 0.2983 - val_loss: 0.2990 - val_mean_squared_error: 0.2990
    Epoch 39/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.2924 - mean_squared_error: 0.2924 - val_loss: 0.2912 - val_mean_squared_error: 0.2912
    Epoch 40/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2918 - mean_squared_error: 0.2918 - val_loss: 0.2931 - val_mean_squared_error: 0.2931
    Epoch 41/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2920 - mean_squared_error: 0.2920 - val_loss: 0.3006 - val_mean_squared_error: 0.3006
    Epoch 42/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2908 - mean_squared_error: 0.2908 - val_loss: 0.2915 - val_mean_squared_error: 0.2915
    Epoch 43/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 0.2887 - mean_squared_error: 0.2887 - val_loss: 0.3125 - val_mean_squared_error: 0.3125
    Epoch 44/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2884 - mean_squared_error: 0.2884 - val_loss: 0.2837 - val_mean_squared_error: 0.2837
    Epoch 45/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 0.2860 - mean_squared_error: 0.2860 - val_loss: 0.3010 - val_mean_squared_error: 0.3010
    Epoch 46/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2882 - mean_squared_error: 0.2882 - val_loss: 0.3038 - val_mean_squared_error: 0.3038
    Epoch 47/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2875 - mean_squared_error: 0.2875 - val_loss: 0.3055 - val_mean_squared_error: 0.3055
    Epoch 48/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 0.2863 - mean_squared_error: 0.2863 - val_loss: 0.2827 - val_mean_squared_error: 0.2827
    Epoch 49/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2855 - mean_squared_error: 0.2855 - val_loss: 0.2858 - val_mean_squared_error: 0.2858
    Epoch 50/50
    [1m133/133[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2908 - mean_squared_error: 0.2908 - val_loss: 0.2832 - val_mean_squared_error: 0.2832



```python
pd.DataFrame(history.history).plot()
```




    <Axes: >




    
![png](wide_and_deep_nn_files/wide_and_deep_nn_21_1.png)
    


## evaluate


```python
model.evaluate(xtest,ytest)
```

    [1m  1/129[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m3s[0m 25ms/step - loss: 0.4699 - mean_squared_error: 0.4699

    [1m129/129[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2978 - mean_squared_error: 0.2978





    [0.2977759838104248, 0.2977759838104248]



