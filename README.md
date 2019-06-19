# Machine Learning in Action

Author: Pedro Paez
github: [https://github.com/pedrojpaez/dataweek.git](https://github.com/pedrojpaez/dataweek)

In this lab we will be going through the entire Data Science workflow using **Sagemaker**. The objective of this exercise is to build from scratch a Data Science project and to learn how **Sagemaker** helps accelerate the process of building and deploying in production custom machine learning models. We will see how to leverage Sagemaker's first party algorithms as well as the high level SDF for Deep Learning frameworks.

We will be building and end-to-end Natural Language Processing pipeline to classify newspaper headlines into general categories. We will first build word embeddings (vector representations of the english vocabulary) to enrich our model. 



##  Prerequisites:

For this lab you will need to have:

 - A laptop
 - Network connectivity
 - An AWS account
 - Basic Python scripting experience
 - Basic knowledge of Data Science workflow

Preferred knowledge:
 - Basic knowledge of containers
 - Basic knowledge of deep learning




##  Part 1 : Prepare environment and create new Sagemaker project

 1. Go to AWS Console in your account
 2. On the top right corner select region N.Virginia

![enter image description here](https://lh3.googleusercontent.com/v-G3E9ggNbbeHRIXlEfxd4kXIq4aoHDD26hN6sJomu0WCuULD9XIWKCXc0gLWVuMohKbftRP2REK "Console")

 3. Search and click on Amazon Sagemaker
 4. Under Notebook > Select Notebook Instance > and click on "Create Notebook Instance" button (orange button)
 
 ![enter image description here](https://lh3.googleusercontent.com/fpI3YGLlxKkbX05p6-tqgEn9PPorKD-r3VoA9EMWf0vemlzwJVTSb4N3lIVggye8OH6BFvB0Ctce "Sagemaker console")
 
 5. Give your project a name under "Notebook instance name"
 6. Select ml.t2.medium Notebook instance type

![enter image description here](https://lh3.googleusercontent.com/8bgiDtXh71fCGOerv8yORgRwJK1ZCAZOwMKqYtDeJG3Z35tTTtaI1CPQ2J9h8QyP6shyZZtRdEji)

 6. Under "Permissions and encryption" > Under IAM role > select "Create a new role" in the scroll down menu
 ![enter image description here](https://lh3.googleusercontent.com/pcQEJ0PHR0Qr6yiksw-5RA2vepl5jvyOqj2hUW0SVMD-4tW82LN1LAlyX2Mdxl0EA3S0TR5zunn3 "Create Role")
 
 7. Select "Any S3 bucket" > Click on "Create new role" button

![enter image description here](https://lh3.googleusercontent.com/nVB4ET4mTMpVvK7VDdQF2sWJGSIK6ZMWC4uuPh1o2ckvBRSY0ualc23fFjJZjuJeYaZy_AcAtr8n "Permissions")

 8. Finally "Create Notebook Instance" and wait until status is "InService"

![enter image description here](https://lh3.googleusercontent.com/LtlMMtFoOFPeVDrDCP86JR5jPmusFQqrO1nZeIkkiC3FFOkKRNL08TmrcCc6_F1jBqPI2NSH2MUR)

###  Clone git repo with workshop material

 1. Select "Open Jupyter". You should see a Jupyter notebook web interface.
 2. Select "New" in the top right corner > Click on "Terminal". A new tab will open with access to the Shell.

![enter image description here](https://lh3.googleusercontent.com/4iSVe9Mx3jpVKZ90ATCatzGp11HQPz35l1z3f2oD7aJlt_3cwDrcySt9nZU9_WCtlQ8Wy0QFYgnt "New")

 1. You now have shell access to the notebook instance and full control/flexibility over your environment. We will cd (change directory to the Sagemaker home directory). Type from the root directory : `cd Sagemaker`
 2. We will clone the material for this lab from the git repo : https://github.com/pedrojpaez/dataweek.git

    git clone https://github.com/pedrojpaez/dataweek.git
    
    ![enter image description here](https://lh3.googleusercontent.com/Vu79ByoR2GFMrzTs-iweyFc8ocqV7C6o5uKZ9TdrmYxwB5gKVnqOqwjUo_PyC4_n5a9x8HZn_Mfm "git_clone")
    
 3. Return to previous tab (Jupyter notebook web interface). The dataweek directory should now be available.

![enter image description here](https://lh3.googleusercontent.com/C4xvL_DA09IOfJqc7bPbB9w49jtB9wAyT7tZ7hScaK5QHJSNPXZu25mjwLcmsnKYWND2bjBXuOuu)

### dataweek directory
There are 4 elements in the dataweek directory:

 - **tf-src**: This directory contains the MXNet training script for our  document classifier.
 - **blazingtext_word2vec_text8.ipynb**: Notebook to create word embeddings using the Sagemaker first party algorithm Blazingtext. We will use these embeddings as input for our headline classifier to enrich the model.
 - **headline-classifier-local.ipynb**: Notebook to create headline classifier using keras (with MXNet backend) on the local instance.
 - **headline-classifier-mxnet.ipynb**: Notebook to create headline classifier leveraging Sagemaker training and deploying features. We will use MXNet high-level SDK to bring our MXNet code and run and deploy our model.

![enter image description here](https://lh3.googleusercontent.com/OjGqwGCyR2bQ2-7RoIY3SDYcVvz467yIFapVMle3aULUL39lwl0T8auVAsv4dWHQ7tHkabEENn9m)

### Run blazingtext_word2vec_text8.ipynb notebook

In this notebook we will run through the snippets of code. We will be building a word embedding model (vector representations of the english vocabulary) to use as input for our document classification model.

For this notebook we will use the first party algorithm Blazingtext to build our word embeddings and we will leverage the one-click training/one-click deployment capabilities of Sagemaker.

The general actions we will be running:

 1. Configure notebook
 2. Download text8 corpus file
 3. Upload data to S3 
 4. Run training job on Sagemaker
 5. Deploy model
 6. Download model object and unpack wordvectors
 7. Clean up (delete model endpoint)

Run through the notebook and read the instructions.

### Run headline-classifier-local.ipynb notebook

In this notebook we will run through the snippets of code. We will build a headline classifier model that will classify newspaper headlines into 4 classes. We will build a deep learning model using the Keras interface with MXNet backend (and use the word embeddings we previously built as input to our model). We will run the training on locally (on the notebook instance) to evaluate performance.

The general actions we will be running:

 1. Configure notebook
 2. Download NewsAggregator datasets
 3. Upload data to S3 
 4. Run training job locally
 5. Move to the next notebook.


Run through the notebook and read the instructions.

### Run headline-classifier-mxnet.ipynb notebook


In this notebook we will run through the snippets of code. We will build a headline classifier model that will classify newspaper headlines into 4 classes. We will build a deep learning model using the Keras interface with MXNet backend (and use the word embeddings we previously built as input to our model). We will run the training on Sagemaker and package the MXNet code to a training script and we will evaluate performance. Finally we will deploy our model as a RESTful API.

The general actions we will be running:

 1. Configure notebook
 3. Upload data to S3 
 4. Run training job on Sagemaker
 5. Deploy model on Sagemaker
 7. Clean up (delete model endpoint)

Run through the notebook and read the instructions.

### Things to try at home

 -Invoke a model endpoint deployed by Amazon SageMaker using API Gateway and AWS Lambda for additional functionality.
https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/


-Analyze the results of your model responses to real time data (for this switch the Comprehend API for your Sagemaker endpoint API).
[https://aws.amazon.com/blogs/machine-learning/build-a-social-media-dashboard-using-machine-learning-and-bi-services/](https://aws.amazon.com/blogs/machine-learning/build-a-social-media-dashboard-using-machine-learning-and-bi-services/)

