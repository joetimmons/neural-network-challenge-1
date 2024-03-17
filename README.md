# neural-network-challenge-1

## Student Loan Risk Prediction with Deep Learning
This project demonstrates the use of deep learning to predict the risk of student loan repayment. It utilizes a neural network model built with TensorFlow's Keras to classify whether a student is likely to successfully repay their loan based on various features.

## Dataset
The project uses the "student-loans.csv" dataset, which contains information about students and their loan repayment behavior. The dataset includes features such as payment history, location parameter, STEM degree score, GPA ranking, alumni success, study major code, time to completion, finance workshop score, cohort ranking, total loan score, financial aid score, and credit ranking.

## Requirements
To run this project, you need the following dependencies:

Python 3.x
pandas
TensorFlow
scikit-learn
## You can install the required packages using pip:
pip install pandas tensorflow scikit-learn

## Project Structure
The project consists of the following sections:

Prepare the data to be used on a neural network model
Read the "student-loans.csv" file into a Pandas DataFrame.
Create the features (X) and target (y) datasets based on the preprocessed DataFrame.
Split the features and target sets into training and testing datasets.
Use scikit-learn's StandardScaler to scale the features data.
Compile and Evaluate a Model Using a Neural Network
Create a deep neural network using TensorFlow's Keras, specifying the number of input features, layers, and neurons.
Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
Evaluate the model using the test data to determine the model's loss and accuracy.
Save and export the trained model to a Keras file named "student_loans.keras".
Predict Loan Repayment Success by Using the Neural Network Model
Reload the saved model.
Make predictions on the testing data and save the predictions to a DataFrame.
Display a classification report with the true labels and predicted labels.
Discuss creating a recommendation system for student loans
Describe the data needed to build a recommendation system for student loan options.
Discuss the choice of filtering method (collaborative, content-based, or context-based) based on the selected data.
Describe two real-world challenges to consider while building a recommendation system for student loans.

## Usage
Make sure you have the required dependencies installed.
Download the "student-loans.csv" dataset and place it in the same directory as the Jupyter Notebook.
Open the Jupyter Notebook and run the cells in sequential order.
The trained model will be saved as "student_loans.keras" in the same directory.
The classification report and discussion on creating a recommendation system will be displayed in the notebook.
## Results
The trained neural network model achieves an accuracy of around 74% on the test data, indicating its ability to predict student loan repayment success with reasonable accuracy.

The classification report provides insights into the model's performance, including precision, recall, and F1-score for each class (successful repayment and unsuccessful repayment).

## Future Work
To further improve the model's performance and create a comprehensive recommendation system for student loans, consider the following:

Collect additional relevant data as described in the discussion section to enhance the recommendation system.
Experiment with different neural network architectures, hyperparameters, and optimization techniques.
Explore other machine learning algorithms and compare their performance with the current model.
Develop a user-friendly interface for students to input their information and receive personalized loan recommendations.
Feel free to contribute to this project by submitting pull requests or opening issues for any suggestions or improvements.
