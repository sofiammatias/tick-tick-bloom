# Tick-Tick-Bloom Challenge from DrivenData

![image](https://drivendata-public-assets.s3.amazonaws.com/competition_cyano_banner.jpeg)

The goal of this challenge was to use satellite imagery to detect and classify the severity of cyanobacteria blooms in small, inland water bodies. The resulting algorithm will help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs.

A significant challenge water quality managers face is the formation of harmful algal blooms (HABs). HABs produce toxins that are poisonous to humans and their pets, and threaten marine ecosystems by blocking sunlight and oxygen. One of the major types of HABs is cyanobacteria. Manual water sampling, or “in situ” sampling, is generally used to monitor cyanobacteria in inland water bodies. In situ sampling is accurate, but time intensive and difficult to perform continuously.

With this model and sattelite imagery from a platform like Microsoft's [Planetary Computer](https://planetarycomputer.microsoft.com/docs/overview/about) will make this work much easier and less expensive.

See more about the challenge description [here](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/649/)

**Ranked #62 among 1,377 participants.**

## Objective

The objective of this challenge was to obtain the **severity** of HAB's, namely cyanobacteria's density, in each water body location of USA, by using sattelite imagery. Through the analysis of one or more images per location, the predictive model will supply the severity value, making **severity** the target of this model. The locations were divided by region and this was also an important features of this problem.

This becomes, therefore, a machine learning multiclassification problem, where the severity classes are represented by numbers: from 1 (least severe density) to 5 (most severe density). 

The model was evaluated by an average RMSE value by region. 

## Data

Data features for this problem were to be obtained by API's, no data was provided directly by the challenge. As I've started this challenge late and hadn't much time, I opt to get images from Sentinel and Landsat only, with the help of the challenge benchmark blog. Still, the planetary computer API is quite easy to use with a good tutorial available [here](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/).

There was a metadata file with geograpical location and a measure of density and severity for each location, which would become the training model. The test data was also provided with all train infomation, except the severity values.

Again, data was well organized and it was easy to start the challenge without too much time spent in gathering data, even if it took several hours to download data from the planetary computer servers.
A routine, taken from the benchmark blog, was implemented to save files locally. Even if the routine was loading all the files each time it run, it was checking first if files were saved locally, and would load them from disk before downloading again. Loading files from disk was fast enough, and even if I was using Google Colab that has a lot of interruptions with runtime, I didn't have troubles in getting all data within one day.

Bear in mind that image size and other image parameters were quite important and heavily affected downloading time. Therefore, using a lot of resolution to improve the model results was really not an option, or even iterating different sorts of bands.

# Exploratory Data Analysis

About EDA, there was a lot of it done on the benchmark blog, and I took advantage of it :) I've added my own graphic to the set, as I ended up finding out a relation between the target classes (severity) and the region:

<img width="641" alt="image" src="https://user-images.githubusercontent.com/114782592/223540460-6de03be9-7d4e-4b8f-ba54-2c095fa8ac82.png">

This graphic shows **class imbalance** from region to region, something I've found out by persistently observing one region RMSE much higher than the rest.
It was when I applied a weight to each of these classes to the model I was using that I could really surpass the benchmark score, from 1.57 to 1.2

# Model

When further studying the benchmark blog, I learned about a new machine learning model, which is an improvement of gradient boosting and decision tree models: LightGBM (Light Gradient Boosting Machine). LGBM is a gradient boosting framework that uses tree based learning algorithms. Find out all about this model [here](https://lightgbm.readthedocs.io/en/latest/index.html).

![LGBM](https://lightgbm.readthedocs.io/en/v3.3.2/_images/LightGBM_logo_black_text.svg)

This model is simple and easy to implement, yet it displays consistently good results for a machine learning model. It has the big advantage of **not needing to preprocess categorical data** (with one-hot encoding or others) and it does the job for you, as long as you do a little trick: change the dtype for each categorical column to "category" in your Python dataframe.

It has a great deal of parameters you can fine-tune and it is a pain to do it without a automated optimizer, like Optuna. This was the work done in another challenge ([check the notebook](https://www.kaggle.com/code/sofiamatias/spaceship-titanic)). 

Still, the big breakthrough was to use class weights as one of the model hyperparameters, although the weight values calculation was not that straightforward, but I've ended up using number of rows for each class / total number of rows.

# Conclusions

This challenge had several aspects to look for:
* Using images as features 
* Loading data with an API
* Using categorical+numerical features
* Class imbalance
* A newfound machine learning model

There were several other aspects to look for, like using properly all features presented (I am still unable to use features for training that are not in a test set), using other images, using other image parameters (like different bands, numerical relations between them) or even using other machine learning techniques to retrieve information from an image. 

I still had time to build and train a convolutional neural network, but it didn't beat the LGBM performance, either because I am too green to use neural networks or because of the nature of data to be processed. I had an opportunity to learn in this challenge about LGBM package, how to deal with class imbalance, and how to use LGBM with categorical features without any preprocessing (LGBM deals with categorical features on its own).



