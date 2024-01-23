### The Eyeball Gang

For our model we tried multiple approaches but in the end nothing was really that successful, so here we are presenting you something which is simple to run and for us to package/send to you to reduce the chances of us making a mistake. 

Model description:
We created a simple Linear Regression model which predicts patient Age from the image represented as a 3 element vector, where each element is the average value of the R,G,orB color channel respectively. 

Preprocessing:
we take each image and cropped it down to the eye, and then generate the above mentioned vector from the cropped image. 
data was stored in a pandas dataframe with values 'Age', 'R', 'G', 'B'. 

Model training:
we just used sklearn for the linear regresion model and spliting the training / test data

Model evaluation:

how to use:
The code is well documented but essentially you have to load 3 main files into your editor (we recomend Spyder):
main.py, data_generator.py, model_generator.py. We provide the already generated dataset so you don't have to use data_generator.py to do so. if you want to generate the data yourself, see the data_generator.py file and then look at the model_generator.py file after. this will require you to copy all the images into the images folder. If you don't want to generate the data yourself, then just run the main.py file. We assume you want to evaluate just a single image like in the hackathon meeting. So in the project folder (same level as the files i previously mentioned) simply drag in the image you want to evaluate and then select that image (see main.py). It should run without issue.


