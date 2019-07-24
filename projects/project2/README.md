---
layout: post
title: Want to avoid crash acccident - Don't go to high-crime neighborhoods
---

![Image test](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_Chicago-Car.jpg)


My approach
-----------
The goal for the second project is to analysis and predict the number of car accidents in the city of Chicago, taking into account of multiple variables such as crime rate, weather, accident location , driving conditions, population density, traffic density and road condition. 





Table of content

- [Get the data](#misc1)

- [Exploratory Data Analysis](#misc2)

- [Model selection](#misc3)

- [Improvements](#misc4)

<!---Table of content

- Get the data

- Exploratory Data Analysis

- Model selection

- Improvements  --->



Packages:
-----------

- `Pandas`: for data frames
- `Seaborn` : for statistical data visualization
- `Statsmodels` : for statistical models, and perform statistical tests
- `Scikit-learn` : for machine learning library
- `Scipy` : for mathematical functions.
- `dbfread` : for reading DBF files and returning the data as native Python data types
- `Shapely` : for manipulation and analysis of planar geometric objects
- `Folium` : for interactive map


<h2 id="misc1">Get the data</h2> 

<!--- Get the data 
----------- --->


Now that we know what we want to predict, what are the inputs? What could cause a car crash. There are many factors, some of which I include in this analysis.


- Weather (temperature, wind speed, visibility, rainy/snowy/icy, snow depth, rain depth, etc)

- Time features: hour of day, day of the week, month of the year.

- Static features such as numbero of intersectiosn , speed limit, average traffic volumes, etc

- Human factors such as population density, etc

- Countless others


The city of Chicago publishes those dataset that are worth exploring:

- [Chicago Traffic Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/data)

- [Chicago Crimes Rate](https://data.cityofchicago.org/Public-Safety/Crimes-One-year-prior-to-present/x2n5-8w5q/data)

- [Chicago Potholes](https://data.cityofchicago.org/Transportation/Potholes-Patched-Previous-Seven-Days/caad-5j9e?referrer=embed)

- [US Zip Code & Population](https://simplemaps.com/data/us-zips)

- [Chicago GeoJSON](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-ZIP-Codes/gdcf-axmw)


 I find the vehicle accidents to be strongly related to location (Geography), so I will grab the Geojson file, so that we can perform geographic data analysis and aggregate the data at zipcode level and month level.


#### Handling with spatial data to geoinference the zipcode

I utilized dataset from [OSMnx Street Networks Dataverse - U.S. Street Network Shapefiles, Node/Edge Lists, and GraphML Files](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CUWWYJ) which provides the lat/lon coordinates of all crossroads in Chicago. There are total 28372 intersections recorded in Chicago. The next step is to retrieve the zipcode for each crossroad.


Given geoJSON database of Chicago with lots of polygons (census tracts specifically) and intersection coordinates, I did some geoinferencing with Shapely and GeoJSON to identify which census tract and zipcode each intersection belongs to.


step 1. Import all the required libraries
```python
import pandas as pd
import json
from shapely.geometry import shape, Point
```


step 2. Load the dataset and geoinference the zipcode
``` python
# load GeoJSON file containing sectors
with open('./dataset/chicago_boundaries_zipcodes.geojson') as f:
    js = json.load(f)



# load file containing lat/lon coordinate of each intersection
df_intersection = pd_read_csv('./dataset/intersections.csv')

# extract the zipcode from coordinates
zipcode = []
for i in range(len(df_intersection)):

    # construct point based on lon/lat returned by geocoder
    point = Point( df_intersection.iloc[i] ['LONGITUDE'], df_intersection.iloc[i] ['LATITUDE'] )


    # check each polygon to see if it contains the point
    for feature in js['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            #print('Found containing polygon:', feature['properties']['zip'])
            zipcode.append(feature['properties']['zip'])
```

![crash](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_crash.png)
<!--- ![pothole](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_pothole.png) --->
<!--- ![population](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_population.png) --->
<!--- ![intersection](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_intersection.png) --->
<!--- ![crime](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_crime.png) --->
<iframe src="public/html/chicago_map.html" height="500px" width="100%"></iframe>



<h2 id="misc2">Exploratory Data Analysis</h2> 
<!--- Exploratory Data Analysis
----------- --->

By exploring the data and analyzing the vehicle crash and other factors , we noticed there are some correlations in dataset. At first, let's check the relationship between number of crossroads and number of accidents

![intersection_vs_accident](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_intersection_vs_accident.png)


### Explore the target data and plot histogram of dependent variable to check if it is normal distribution

```
plt.figure(figsize=(12, 8))
sns.distplot(df['num_accident'])
plt.title('Histogram of target')
```
![pothole](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_target_hist.png)




### Pairwise scatter plots and correlation heatmap for checking multicollinearity

![pairplot](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_pairplot.png)




### Visualize Confusion Matrix using Heatmap
Here, I visualize the confusion matrix using Heatmap.

![correlationmatrix](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_correlationmatrix.png)


I anticipated there is a strong correlation between condition road and car crash. Contrary to expectations, the crime rate turned out to positively affect the accident rate. 



<h2 id="misc3">Model selection</h2> 
<!--- Model Selection
----------- --->

Based on the exploratory analysis, here are the features I use as independent variables to fit model :

- `percent_accident_at_night` -- percentage of accident at night 
- `percent_accident_in_snow`  -- percentage of accident in snow 
- `lat`, `lgn` -- latitude and longitude of each zipcode
- `population` -- population of zipcode where car accident happens
- `density`  -- population density per zipcode 
- `num_potholes` -- total number of potholes per zipcode
- `percent_traffic_related_crime` -- percentage of vehicle-related crime
- `num_crime`  -- total number of crimes per zipcode
- `intersection_count` -- number of intersections in each zipcode


When picking a model to fit the data to there are several factors to consider:

- What type of problem am I solving? Classification? Predictive? 

- Does only predictive accuracy matter to me or do I want to also understand how the variables effect the model?

For this problem, I am predicting the number of accidents, which is a classic predictive problem. I do want to understand variables' importance and know how the they effect the model. Ideally, I’d want to tell drivers which area they should avoid and take caution when driving through.   



### Build linear regression model

```python
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
    
train_score = lr_model.score(X_train, y_train)
# score fit model on validation data
val_score = lr_model.score(X_val, y_val)
    
# report results
print('\nTrain R^2 score was:', train_score)
print('\nValidation R^2 score was:', val_score)
```

Here is the $R^2$ score :
```
Train R^2 score was: 0.6717622381724969
Validation R^2 score was: 0.6752109416359127
```



Let's do some feature engineering to check if we can improve $R^2$  

```python
X2 = X.copy()
X2['population2'] = X2['population'] ** 2
X2['num_crime_/_num_intersection'] = X2['num_crime'] / X2['intersection_count']
X2['intersection_count5'] = X2['intersection_count'] ** 5
X2['num_crime_x_num_pothole'] = X2['num_crime'] * X2['num_potholes']
```


Also let's check the accuracy of the model again.

```
Train R^2 score was: 0.7465677013645713
Validation R^2 score was: 0.7505576728834923
```

Surprisingly, $R^2$ score improved significantly which is considered quite good.

I also fitted polynomial model with data and the result is much better

```
Degree 2 polynomial regression val R^2: 0.814
Degree 2 polynomial regression test R^2: 0.812
```

### Check the quality of regression model by plotting residual and prediction

Plot of prediction vs. residuals helps to check homoscedasticity. When we plot the prediction  vs. the residuals, we clearly observe that the variance of the residuals are distributed uniformly randomly around the zero x-axes and do not form specific clusters. Therefore, it respects homoscedasticity.

![pred_res](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_pred_res.png)


### Plot Cooks Distance to detect outliers

```python
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(model_2_fit, alpha  = 0.05, ax = ax, criterion="cooks")
```

![cook_dist](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_cook_dist.png)



### Find the most significant features with LARS model selection

LARS is a method for finding/computing the entire regularization path in a way that is nearly as computationally efficient as lasso because it shrinks the coefficients towards zero in a data-dependent way. Obviously, `num_crime` has the most positive effect on target `num_accident` 


![lars](https://raw.githubusercontent.com/dvu4/dvu4.github.io/master/public/images/p2_lars.png)




### Predict number of accidents for zipcode 60616 (Chinatown and Armour Square)

```python
# Predict car accident at zipcode 60616 in December
crash_id = X_test.index[10]

crash = df['id'].loc[crash_id]
zipcode, v = crash[0], crash[1]

predictions = model_fit.predict(sm.add_constant(X_test))

print(f'The actual number of accidents  at zipcode {zipcode : .0f} in {month : .0f} is {y_test.num_accident[crash_id] : .0f}')
print(f'The predicted number of accidents  at zipcode {zipcode1 : .0f} in {month : .0f} is {predictions[crash_id]: .0f}')
```


```
The actual number of accidents at zipcode  60616 in December is  187
The predicted number of accidents at zipcode  60616 in  December is  162
```

The predicted accidents in Chinatown during December is not too far from the actual one.


<h2 id="misc4">Conclusion</h2> 
<!--- Conclusion
----------- --->


In this blog, I covered some methods to extract spatial data, cleaning and fitting the regression model to estimate the car crash based on the zipcode. There is much room for improvements. 

- Deploying different algorithms to improve $R^2$ score and the accuracy of model. 


- Adding more features such as ridesharing data or median income to figure out the affect of those new features on model.


- Prediting car crash can also reduce the risk through enforcement, education. 


You can find the project on [Github](https://github.com/dvu4/metis_submission/blob/master/projects/project1/project1.ipynb) and reach me out on [Linkedin](https://www.linkedin.com/in/ducmvu/)

Thank you for reading this article.


<!--- https://gist.github.com/VEnis/7465176 --->
