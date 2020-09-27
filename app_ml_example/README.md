# HR predictions


The problem is predict if a employee will left the company knowing the following
features:
- satisfaction_level: satisfaction level
- last_evaluation: score in the last evaluation 
- number_project: project ID that the employee is working
- average_montly_hours: number of hours spent in the project 
- time_spend_company: time that is working in the company 
- Work_accident: number of work accidents
- *left*: predict if a employee will left the company
- promotion_last_5years: identifies if an employee received a promotion 
- sales: department name in which the employee is working
- salary: level of salary: low, medium

The dataset can be download [here](https://www.kaggle.com/jacksonchou/hr-data-for-analytics).


The main purpose is to show how to use FastAPI to deploy a model. Thus, at this moment, we are not
concerning about model's accuracy. To predict if a employee can left the company
we are using a LightGBM model.

The one-hot-encoding is used to encode the categorical features: sales, salary.


# FastAPI

The main file inside of the folder **app** will read the files generated
by the **train.py**. A **class BaseModel** is to track the variables that we use
and to create the documentation based in the class. 


To run the application write in terminal:
- chmod +x bin/start_server: to give permission to execute the bash file
- bin/start_server: to start server
- python exp_app.py: to test the model

To give you more details, to run the uvicorn server you need to be located
in the folder of the project (app_ml_example). The command is: 

- **uvicorn app.main:app** -> it calls the app variable that is inside of main.py file which is
stored in app folder.


# Docker Container

It allow us run the code in all machines without concerns of dependencies once
those will be installed inside of a docker container. Because the docker installation
depends of the Operate System (OS) its better to search how you cna install Docker for
your OS. The first step is create a Docker file.



- docker build --file Dockerfile --tag fastapi-ml .
- docker run -p 8000:8000 fastapi-ml
- docker exec -it cointainer_id bash : if you want to access to docker
        
        - uvicorn app.main:app
        
        - exit to bash: ctrl+d




**[For more details](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-dc51200fe8cf)**

**[FastAPI example](https://github.com/cosmic-cortex)**

**[Docker Introduction](https://blog.boltops.com/2018/04/19/docker-introduction-tutorial)** 
