# biz_stats_ml
Program that produces retail/wholesale trade statistics using machine learning + visualisations 

Code has been moved from my organisation to my personal profile. All of the work was done in the repo that my organisation owns - but all the work is my own. I am sharing here to open it up to comment from people working with similar statistics worldwide. Please feel free to share your thoughts.

The main problem being solved, is inaccuarte/low quality responses being delivered by respondents to a survey - often with very material consequences to the final production. This normally takes a team of statisticians an entire year to correct (sometimes results in having to recontact the responders) - I am to solve this task using machine learning and other statistical measures. 

Results: A full production run (normally completed by a team of 5-7 people over an entire year) completed in 600.38 seconds. Results pass several logical tests and when backtested against former productions compares very favorably. The r^2 when comparing what this program produces against what was actually published was approx 98% with a mean absolute error of approx 5.000 nok - which is low given the characteristics of our data. 

The code uses confidential data and will not run simply by cloning this repo. But I will demonstrate what the code is doing here in this ReadMe file:

## Visualisations:

Several visualisations are used to analyse the data on an industry level. The plots are interactive, the user can select years, chart-types, focus variables etc. All of the regular plotly interactive tools are available as well. Some visualisations are animated and if the user presses play, they will see changes overtime. Here are what some of the outputs look like (which would naturally adjust if something different was selected in the dropdown menus. 

**Simple plots:** 

<img width="990" alt="Line Plots" src="https://github.com/user-attachments/assets/18b5c77b-ee64-4d4a-b79d-7230be50b016"><br><br>




**Bar Charts and Heat Maps:**

<img width="457" alt="Bar Chart and Heat Map" src="https://github.com/user-attachments/assets/3a561afc-0d51-4666-b6a2-ea2860cb0200"><br><br>



**Maps (one that is animated):**


<img width="422" alt="static map" src="https://github.com/user-attachments/assets/6eff24d7-ef9c-43ab-89d1-bf38d0bf87cc">
<img width="485" alt="image" src="https://github.com/user-attachments/assets/181b1f61-79f5-4ae8-87a8-59c7a31b97c5"><br><br>

**Histogram with cumulative percentage:**


<img width="391" alt="histogram" src="https://github.com/user-attachments/assets/35454eef-b4c5-4ae3-9168-30265510f57d"><br><br>


**Linked Plots:**

<img width="394" alt="Linked Plots" src="https://github.com/user-attachments/assets/0f85e171-6b63-493d-9bf9-5fc61dd7c7b4"><br><br>

**Bubble Plots:**

<img width="445" alt="Bubble Plot" src="https://github.com/user-attachments/assets/4e29d2d4-1055-48ce-9d4b-9ff72cdc1e3b"><br><br>

**Parallel Coordinates Plot:**

<img width="845" alt="Parallel Cooridinates" src="https://github.com/user-attachments/assets/2158cb5e-7edf-4262-87a4-b1f5015adff0"><br><br>

**Geographical Plot:**

<img width="851" alt="geographic" src="https://github.com/user-attachments/assets/b367bfe7-9cf0-4874-90ff-3a754b7939eb"><br><br>

**Animated Bar Chart:**


<img width="662" alt="animated bar chat" src="https://github.com/user-attachments/assets/962a2958-4929-4eb2-aa9e-1c79bd7d9d1f"><br><br>

**3D Plot:**

<img width="532" alt="3D" src="https://github.com/user-attachments/assets/b8ae22ca-7f17-421b-bde5-28f379e0edad"><br><br>


## Machine Learning Evaluation

This program aims to solve the problem of low quality financial data surveys responses. We evaluate the quality by comparing responses to skattetateen data and how many of the fields are filled out. Poor quality responses are imputed using a chosen machine learning algorithm which is trained on the full data set (net of poor quality surveys). 

**Important tools used:**

**Feature engineering:** I gathered extra data by quering various apis and cooperating with several other departments within SSB. I also used tools such as KNN imputation to fill NaN values and created new trend variables using Linear Regression. 

**GridSearch:** This was used for hyperparameter tuning. This can be switched on and off depending on the needs of the user. 

**Other key tools and parameters:**

**Scaler (object):** Scalers are used to normalize or standardize numerical features. Common scalers include StandardScaler and RobustScaler. Normalization helps in speeding up the convergence of the training algorithm by ensuring that all features contribute equally to the learning process.

**epochs_number (int):** The number of epochs determines how many times the learning algorithm will work through the entire training dataset. More epochs can lead to better learning but may also result in overfitting if too high. 

**batch_size (int):** This defines the number of samples that will be propagated through the network at one time. Smaller batch sizes can lead to more reliable updates but are computationally more expensive. I chose a medium size number based on the shape of the data, and how often certain features appear within the df. Speed was also a consideration. 

**Early Stopping:** I use early stopping techniques in order to prevent overfitting and improve training time. 

**Learning Curves:** I have used learning curves to determine whether models are overfitting. The results indicate that this has not occurred. 

#### Neural Network Specific Parameters:

**All parameters are subject to change based on results and at times the results of a GridSearch(hyperparameter tuning)**

**learning_rate (float):** In the function, the default learning rate is set to 0.001. The learning rate controls how much the model’s weights are adjusted with respect to the loss gradient. A learning rate of 0.001 is a common starting point as it allows the model to converge smoothly without overshooting the optimal solution.

**dropout_rate (float):** The default dropout rate is set to 0.5. Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of the input units to zero at each update during training. A dropout rate of 0.5 means that half of the neurons are dropped, which is a standard value for promoting robustness in the network.

**neurons_layer1 (int):** The first layer of the neural network has 64 neurons by default. Having 64 neurons allows the model to capture complex patterns in the data while maintaining a balance between computational efficiency and model capacity.

**neurons_layer2 (int):** The second layer has 32 neurons by default. This smaller number of neurons in the subsequent layer helps in reducing the model complexity gradually, which can help in capturing hierarchical patterns in the data.

**activation (str):** The activation function used in the hidden layers is relu (Rectified Linear Unit). The ReLU function is popular because it introduces non-linearity while being computationally efficient and mitigating the vanishing gradient problem common in deeper networks.

**optimizer (str):** The optimizer used is adam by default. Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that has been widely adopted due to its efficiency and effectiveness in training deep neural networks. It combines the advantages of two other extensions of stochastic gradient descent, namely AdaGrad and RMSProp, to provide faster convergence.

**Additional Details on the Model Building Process**

**Layer Construction:**

The first dense layer with 64 neurons uses relu activation, which is ideal for capturing complex non-linear relationships.
A dropout layer follows to prevent overfitting by randomly dropping 50% of the neurons during training.
The second dense layer with 32 neurons also uses relu activation, helping to refine the features extracted by the first layer.
Another dropout layer is added after the second dense layer for additional regularization.
The final output layer has a single neuron with a linear activation function, appropriate for regression tasks as it outputs a continuous value.
Regularization:

The kernel_regularizer=tf.keras.regularizers.l2(0.01) is applied to the dense layers. L2 regularization helps in preventing overfitting by penalizing large weights, thereby promoting smaller, more generalizable weights.

**Results:**

**XGBoost:**

**I used visualisations techniques in order to see the importance of several features. **

<img width="388" alt="XG1" src="https://github.com/user-attachments/assets/5dd15eb7-cd0a-41f9-b81d-4fa9ab640a11">
<img width="395" alt="xg2" src="https://github.com/user-attachments/assets/68290aba-e794-4f60-818c-470d13b78243">
<img width="368" alt="xg3" src="https://github.com/user-attachments/assets/2e10b292-b999-4b14-b465-8e1bb55bb114"><br><br>

**K-Nearest Neighbors:**

<img width="401" alt="NN" src="https://github.com/user-attachments/assets/088dc808-8968-44e7-b2c7-f7a0b47733c8"><br><br>

**Neural Network:**
<img width="446" alt="Neural Networks" src="https://github.com/user-attachments/assets/a34c86cd-704b-4afc-bb80-b9cd8844c085"><br><br>


## DASH APP:

I also created a dashboard using Dash to visualise the final product. Here is a quick snapshot (Theres more), but essentially it is the visualisations seen in the notebook, but in dashboard form where variables can be selected and used to update all plots at once:


<img width="1695" alt="dash 1" src="https://github.com/user-attachments/assets/e70ddcc1-4724-498c-953e-41406d64da42"><br><br>


## Testing the results:

I perform several logical tests and backtest the output of the program against actual publications:

<img width="696" alt="Test Results" src="https://github.com/user-attachments/assets/4fe37337-f077-4d51-b59c-ce6d5b6f0648">

<img width="317" alt="Test Results 2" src="https://github.com/user-attachments/assets/a5bcd20d-9d11-44a9-942a-7503e19b5de5"><br><br>

**Based on these results its likely I will use K-NN nearest neighbors for the 2023 production.**

## Moving Forward:

Models can always be improved. With more resources, particularly time, it may be worth investigating several other opportunities, such as :

- training models for specific industries. Especially if those industries are particularly unique. For example for petrol & diesel sales we can try to use various road network features (distance to nearest gas stations, how often a road is used etc):
- Card transaction data may soon be available, which leads to the possibility of better feature engineering - particularly for retail industries. 
- There is an opportunity to identify which company an industry might belong to, and as a result, identify companies that are currently assigned to the wrong industry (the key for which everything is aggregated). Current classification models perform poorly as seen below. But these only use financial data, I expect if we use features such as job titles (number of employees under a given job title) , then the models will perform better.

**Road Network Data:**

<img width="445" alt="Roads" src="https://github.com/user-attachments/assets/d30ca253-3720-4a19-bc79-4d125bb1f26b"><br><br>


**Classification Performance (So far)**

<img width="287" alt="Classification 1" src="https://github.com/user-attachments/assets/3a03cb33-0d9d-4148-a89d-8e4af063ee27">
<img width="314" alt="Classification 2" src="https://github.com/user-attachments/assets/e1800742-becd-45b3-b725-01519c6312dc">



## Contributers to the project:

<img width="716" alt="contributions" src="https://github.com/user-attachments/assets/7208d70e-c47e-4c16-8ae1-8e51a43b516a">




