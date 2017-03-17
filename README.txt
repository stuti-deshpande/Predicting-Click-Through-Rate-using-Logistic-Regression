Name: Stuti Deshpande

Using Logistic Regression to train the model with the training dataset:
--Optimizing the parameters using Gradient Descent:
Trained the model using LogisticRegressionWithSGD (from pyspark.mllib.classification) with train method by passing parameters as iterations=15, step=1, miniBatchFraction=1, regType=None, validateData="False". Recorded the time to run the Gradient Descent on training dataset which is 110.34 minutes.
--Optimizing the parameters using Stochastic Gradient Descent:
Trained the model using LogisticRegressionWithSGD ( from pyspark.mllib.classification) with train method by passing parameters as iterations=100, step=0.01, miniBatchFraction=0.01, regType=None, validateData="False". Recorded the time to run the Gradient Descent on training dataset which is 148.36 minutes.
-- Obtained the graph of time to train vs. size of the training set to compare both methods, for the best optimized parameters thus obtained.
--Using Logistic Regression with Stochastic Gradient Descent to perform testing on the test dataset.
--Retraining the model using L2 Regularization along with the Stochastic Gradient Descent. accuracy raised up to 0.8341

1. The submit folder has three subfolders: Code, Pseudocode and Output

2. The subfolder Code has four folders:
        1. The folder "Task1-Vectorization" contains python script part1.py for Task-1
        2. The folder "Task-2-Gradient Descent" contains python script part2-gd.py for Task-2 for Gradient Descent
        3. The folder "Task-2-Stochastic-Gradient-Descent" contains python script part2-sgd.py for Stochastic Gradient Descent
        4. The folder "Task3-L2-Regularization" contains python script part3.py for Task-4
        
2. The subfolder Pseudocode has four folders:
        1. The folder "Task1-Vectorization" contains python script part1.txt for Task-1
        2. The folder "Task-2-Gradient Descent" contains python script part2-GD.txt for Task-2 for Gradient Descent
        3. The folder "Task-2-Stochastic-Gradient-Descent" contains python script part2-SGD.txt for Stochastic Gradient Descent
        4. The folder "Task3-L2-Regularization" contains python script part3-L2Regularisation.txt for Task-4
        
3. The subfolder Output has 3 snapshots of output files containing Evaluation Metrics and one graph for the time-to-train VS size-of-file

4. The Report-Assignment3.docx contains the entire documentation of steps and methods followed to do this Assignment, with brief description. 
    It also contains the Outcomes (output for evaluation metrics) and Conclusions drawn.
