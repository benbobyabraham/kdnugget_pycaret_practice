# Pycaret Demo with deployment from kdnuggets tutorial
Reference : [https://www.kdnuggets.com/2020/05/build-deploy-machine-learning-web-app.html](https://www.kdnuggets.com/2020/05/build-deploy-machine-learning-web-app.html)

In our last post we demonstrated how to train and deploy machine learning models in Power BI using PyCaret. If you haven’t heard about PyCaret before, please read our announcement to get a quick start.

In this tutorial we will use PyCaret to develop a machine learning pipeline, that will include preprocessing transformations and a regression model to predict patient hospitalization charges based on demographic and basic patient health risk metrics such as age, BMI, smoking status etc.

 

#### What you will learn in this tutorial
 

- What is a deployment and why do we deploy machine learning models.
- Develop a machine learning pipeline and train models using PyCaret.
- Build a simple web app using a Python framework called ‘Flask’.
- Deploy a web app on ‘Heroku’ and see your model in action.
 

#### What tools we will use in this tutorial?
 
##### PyCaret

PyCaret is an open source, low-code machine learning library in Python to train and deploy machine learning pipelines and models in production. PyCaret can be installed easily using pip.

```
# for Jupyter notebook on your local computer
pip install pycaret# for azure notebooks and google colab
!pip install pycaret
```

##### Flask

Flask is a framework that allows you to build web applications. A web application can be a commercial website, a blog, e-commerce system, or an application that generates predictions from data provided in real-time using trained models. If you don’t have Flask installed, you can use pip to install it.

```
# install flask
pip install Flask
```

##### GitHub

GitHub is a cloud-based service that is used to host, manage and control code. Imagine you are working in a large team where multiple people (sometime hundreds of them) are making changes. PyCaret is itself an example of an open-source project where hundreds of community developers are continuously contributing to source code. If you haven’t used GitHub before, you can sign up for a free account.

##### Heroku

Heroku is a platform as a service (PaaS) that enables the deployment of web apps based on a managed container system, with integrated data services and a powerful ecosystem. In simple words, this will allow you to take the application from your local machine to the cloud so that anybody can access it using a Web URL. In this tutorial we have chosen Heroku for deployment as it provides free resource hours when you sign up for new account.
![Machine Learning Workflow from Training to Deployment on PaaS](https://i.ibb.co/6gK19WD/pycaret-web-app-2.png)

Machine Learning Workflow (from Training to Deployment on PaaS)

#### Why Deploy Machine Learning Models?
 
The deployment of machine learning models is the process of making models available in production where web applications, enterprise software and APIs can consume the trained model by providing new data points and generating predictions.

Normally machine learning models are built so that they can be used to predict an outcome (binary value i.e. 1 or 0 for Classification, continuous values for Regression, labels for Clustering etc. There are two broad ways of generating predictions (i) predict by batch; and (ii) predict in real-time. In our last tutorial we demonstrated how to deploy machine learning model in Power BI and predict by batch. In this tutorial we will see how to deploy a machine learning model to predict in real-time.


#### Business Problem
 
An insurance company wants to improve its cash flow forecasting by better predicting patient charges using demographic and basic patient health risk metrics at the time of hospitalization.
![](https://i.ibb.co/Hn1953X/pycaret-web-app-3.png)

#### Objective
 
To build a web application where demographic and health information of a patient is entered in a web form to predict charges.

 

#### Tasks
 

- Train and validate models and develop a machine learning pipeline for deployment.
- Build a basic HTML front-end with an input form for independent variables (age, sex, bmi, children, smoker, region).
- Build a back-end of the web application using a Flask Framework.
- Deploy the web app on Heroku. Once deployed, it will become publicly available and can be accessed via Web URL.
 

#### Task 1 — Model Training and Validation
 
Training and model validation are performed in Integrated Development Environment (IDE) or Notebook either on your local machine or on cloud. In this tutorial we will use PyCaret in Jupyter Notebook to develop machine learning pipeline and train regression models. If you haven’t used PyCaret before, click here to learn more about PyCaret or see Getting Started Tutorials on our website.

In this tutorial, we have performed two experiments. The first experiment is performed with default preprocessing settings in PyCaret (missing value imputation, categorical encoding etc). The second experiment has some additional preprocessing tasks such as scaling and normalization, automatic feature engineering and binning continuous data into intervals. See the setup example for the second experiment:

```
# Experiment No. 2from pycaret.regression import *r2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])
```

![Comparison of information grid for both experiments](https://i.ibb.co/Qcy6R9M/pycaret-web-app-4.png)

Comparison of information grid for both experiments

The magic happens with only a few lines of code. Notice that in Experiment 2 the transformed dataset has 62 features for training derived from only 7 features in the original dataset. All of the new features are the result of transformations and automatic feature engineering in PyCaret.

![Columns in dataset after transformation](https://i.ibb.co/rZRGQtV/pycaret-web-app-4-b.png)

Columns in dataset after transformation

 

Sample code for model training and validation in PyCaret:

```
# Model Training and Validation 
lr = create_model('lr')
```

![](https://i.ibb.co/0BFw93Y/pycaret-web-app-5.png)

10 Fold cross-validation of Linear Regression Model(s)

Notice the impact of transformations and automatic feature engineering. The R2 has increased by 10% with very little effort. We can compare the residual plot of linear regression model for both experiments and observe the impact of transformations and feature engineering on the heteroskedasticity of model.

```
# plot residuals of trained model
plot_model(lr, plot = 'residuals')
```
![](https://i.ibb.co/yfMy3Z9/pycaret-web-app-6.png)

Residual Plot of Linear Regression Model(s)

Machine learning is an iterative process. Number of iterations and techniques used within are dependent on how critical the task is and what the impact will be if predictions are wrong. The severity and impact of a machine learning model to predict a patient outcome in real-time in the ICU of a hospital is far more than a model built to predict customer churn.

In this tutorial, we have performed only two iterations and the linear regression model from the second experiment will be used for deployment. At this stage, however, the model is still only an object within notebook. To save it as a file that can be transferred to and consumed by other applications, run the following code:

```
# save transformation pipeline and model 
save_model(lr, model_name = 'c:/username/ins/deployment_28042020')
```

When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the setup() function is created . All inter-dependencies are orchestrated automatically. See the pipeline and model stored in the ‘deployment_28042020’ variable:

![](https://i.ibb.co/XDyHLLp/pycaret-web-app-7.png)

Pipeline created using PyCaret


We have finished our first task of training and selecting a model for deployment. The final machine learning pipeline and linear regression model is now saved as a file in the local drive under the location defined in the save_model() function. (In this example: c:/username/ins/deployment_28042020.pkl).

 
#### Task 2 — Building Web Application
 
Now that our machine learning pipeline and model are ready we will start building a web application that can connect to them and generate predictions on new data in real-time. There are two parts of this application:

- Front-end (designed using HTML)
- Back-end (developed using Flask in Python)

#### Front-end of Web Application
 
Generally, the front-end of web applications are built using HTML which is not the focus of this article. We have used a simple HTML template and a CSS style sheet to design an input form. Here’s the HTML snippet of the front-end page of our web application.

![](https://i.ibb.co/7yhsQsV/pycaret-web-app-8.png)

Code snippet from home.html file

You don’t need to be an expert in HTML to build simple applications. There are numerous free platforms that provide HTML and CSS templates as well as enable building beautiful HTML pages quickly by using a drag and drop interface.

##### CSS Style Sheet

CSS (also known as Cascading Style Sheets) describes how HTML elements are displayed on a screen. It is an efficient way of controlling the layout of your application. Style sheets contain information such as background color, font size and color, margins etc. They are saved externally as a .css file and is linked to HTML but including 1 line of code.

![](https://i.ibb.co/mGy3tCL/pycaret-web-app-9.png)

Code snippet from home.html file

#### Back-end of Web Application
 
The back-end of a web application is developed using a Flask framework. For beginner’s it is intuitive to consider Flask as a library that you can import just like any other library in Python. See the sample code snippet of our back-end written using a Flask framework in Python.

![](https://i.ibb.co/71hvBj0/pycaret-web-app-10.png)

Code snippet from app.py file


If you remember from the Step 1 above we have finalized linear regression model that was trained on 62 features that were automatically engineered by PyCaret. However, the front-end of our web application has an input form that collects only the six features i.e. age, sex, bmi, children, smoker, region.

How do we transform 6 features of a new data point in real-time into 62 features on which model was trained? With a sequence of transformations applied during model training, coding becomes increasingly complex and time-taking task.

In PyCaret all transformations such as categorical encoding, scaling, missing value imputation, feature engineering and even feature selection are automatically executed in real-time before generating predictions.

Imagine the amount of code you would have had to write to apply all the transformations in strict sequence before you could even use your model for predictions. In practice, when you think of machine learning, you should think about the entire ML pipeline and not just the model.

Testing App

One final step before we publish the application on Heroku is to test the web app locally. Open Anaconda Prompt and navigate to folder where ‘app.py’ is saved on your computer. Run the python file with below code:

```
python app.py
```

![](https://i.ibb.co/xDngYdg/pycaret-web-app-11.png)

Output in Anaconda Prompt when app.py is executed

 

Once executed, copy the URL into a browser and it should open a web application hosted on your local machine (127.0.0.1). Try entering test values to see if the predict function is working. In the example below, the expected bill for a 19 year old female smoker with no children in the southwest is $20,900.

![](https://i.ibb.co/S7SgKdR/pycaret-web-app-12.png)
Web application opened on local machine

 

Congratulations! you have now built your first machine learning app. Now it’s time to take this application from your local machine into the cloud so other people can use it with a Web URL.

 
#### Task 3 — Deploy the Web App on Heroku
 
Now that the model is trained, the machine learning pipeline is ready, and the application is tested on our local machine, we are ready to start our deployment on Heroku. There are couple of ways to upload your application source code onto Heroku. The simplest way is to link a GitHub repository to your Heroku account.

If you would like to follow along you can fork this repository from GitHub. If you don’t know how to fork a repo, please read this official GitHub tutorial.

![](https://i.ibb.co/brqDkWQ/pycaret-web-app-13.png)
[https://www.github.com/pycaret/deployment-heroku](https://www.github.com/pycaret/deployment-heroku)

 

By now you are familiar with all the files in repository shown above except for two files i.e. ‘requirements.txt’ and ‘Procfile’.

![](https://i.ibb.co/wgDJSR9/pycaret-web-app-13-b.png)
requirements.txt

 

requirements.txt  file is a text file containing the names of the python packages required to execute the application. If these packages are not installed in the environment application is running, it will fail.

![](https://i.ibb.co/f1Tw86N/pycaret-web-app-13-c.png)
Procfile

 

Procfile is simply one line of code that provides startup instructions to web server that indicate which file should be executed first when somebody logs into the application. In this example the name of our application file is ‘app.py’ and the name of the application is also ‘app’. (hence app:app)

Once all the files are uploaded onto the GitHub repository, we are now ready to start deployment on Heroku. Follow the steps below:

Step 1 — Sign up on heroku.com and click on ‘Create new app’

![](https://i.ibb.co/YLy4htZ/pycaret-web-app-14.png)
Heroku Dashboard

 

Step 2 — Enter App name and region

![](https://i.ibb.co/JsnqrVn/pycaret-web-app-15.png)
Heroku — Create new app

 

Step 3 — Connect to your GitHub repository where code is hosted

![](https://i.ibb.co/L6HWPcq/pycaret-web-app-16.png)
Heroku — Connect to GitHub

 

Step 4 — Deploy branch

![](https://i.ibb.co/YtkPLbk/pycaret-web-app-17.png)
Heroku — Deploy Branch

 

Step 5 — Wait 5–10 minutes and BOOM

![](https://i.ibb.co/YtkPLbk/pycaret-web-app-18.png)
Heroku — Successful deployment

 

App is published to URL: [https://pycaret-insurance.herokuapp.com/](https://pycaret-insurance.herokuapp.com/)

![](https://i.ibb.co/F6hnZRT/pycaret-web-app-19.png)
[https://pycaret-insurance.herokuapp.com/](https://pycaret-insurance.herokuapp.com/)

 

There is one last thing to see before we end the tutorial.

So far we have built and deployed a web application that works with our machine learning pipeline. Now imagine that you already have an enterprise application in which you want to integrate predictions from your model. What you need is a web service where you can make an API call with input data points and get the predictions back. To achieve this we have created the predict_api function in our ‘app.py’ file. See the code snippet:

![](https://i.ibb.co/NFtjFxk/pycaret-web-app-20.png)
Code snippet from app.py file (back-end of web app)

 

Here’s how you can use this web service in Python using the requests library:

```
import requestsurl = 'https://pycaret-insurance.herokuapp.com/predict_api'pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})print(pred.json())
```

![](https://i.ibb.co/hdK5PWR/pycaret-web-app-21.png)
Make a request to a published web service to generate predictions in a Notebook

 

 

Next Tutorial
 
In the next tutorial for deploying machine learning pipelines, we will dive deeper into deploying machine learning pipelines using docker containers. We will demonstrate how to easily deploy and run containerized machine learning applications on Linux.
