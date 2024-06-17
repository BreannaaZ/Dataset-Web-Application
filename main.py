import streamlit as st
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ---------------------- DATA SETUP --------------------- #
# Load the dataset
def load_data():
    return pd.read_csv('Dataset/diabetes.csv')


data = load_data()

# Clean the data
# Remove rows with empty information
data.dropna(inplace=True)

# Filter columns (based on actually possible measurements)
data = data[(data['Pregnancies'] >= 0) &
            (data['Age'] >= 21) &
            (data['Age'] <= 100) &
            (data['Glucose'] >= 20) &
            (data['Glucose'] <= 600) &
            (data['BloodPressure'] >= 0) &
            (data['BloodPressure'] <= 300) &
            (data['SkinThickness'] >= 0) &
            (data['SkinThickness'] <= 50) &
            (data['Insulin'] >= 0) &
            (data['BMI'] >= 10) &
            (data['BMI'] <= 100) &
            (data['DiabetesPedigreeFunction'] >= 0) &
            (data['DiabetesPedigreeFunction'] <= 2)  # Adjust this range as needed
            ]

# Reset index after filtering
data.reset_index(drop=True, inplace=True)

# Save the cleaned dataset
data.to_csv('cleaned_diabetes.csv', index=False)

# ----------- WEB APP SECTIONS ---------------- #

# Title of the web app
st.title('CIS 335 Project by Breanna Zinky')

# Sub title
st.write("""
## Diabetes Dataset Experimentation App
""")

# Other text to be displayed
st.write(f'This StreamLit Web App allows you to experiment with different types of normalization and classification on '
         f'the given diabetes dataset.')
st.write(f'This diabetes dataset includes information that can be used to predict whether a patient has diabetes based '
         f'on diagnostic measurements')
st.write(f'This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. All '
         f'patients included in the dataset are females, at least 21 years old, of Pima Indian heritage.')
st.write(f'Go ahead and explore it:')

# Create two columns
left_column, right_column = st.columns(2)

# Display dataset in the left column
with left_column:
    # Display the dataset
    st.write('## Diabetes Dataset')
    st.write(data)

# ------------------------ NORMALIZATION INTERACTIVITY ------------------------ #

# Sidebar for normalization options
st.sidebar.title("Normalization Options")
normalization_option = st.sidebar.selectbox("Select normalization method:",
                                            ["No Normalization", "Z-score Normalization", "Min-Max Normalization"],
                                            index=0)

# Define columns to be normalized (excluding 'Outcome')
columns_to_normalize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age']

# Normalization logic based on user selection
if normalization_option == "No Normalization":
    normalized_data = data.copy()
elif normalization_option == "Z-score Normalization":
    normalized_data = data.copy()
    for column in columns_to_normalize:
        if column != 'Outcome':
            normalized_data[column] = (data[column] - data[column].mean()) / data[column].std()
elif normalization_option == "Min-Max Normalization":
    normalized_data = data.copy()
    for column in columns_to_normalize:
        if column != 'Outcome':
            normalized_data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

# Display normalized chart in the right column
with right_column:
    # Display normalized data
    st.write('## Normalized Dataset')
    st.write(normalized_data)

# --------------------- DISPLAY SOME OTHER INFO ABOUT DATASET ----------------- #

st.write('## Inspect the Dataset')

# Allow the user to check a box to display different information about the dataset
# Column names and descriptions
column_descriptions = {
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin fold thickness (mm)",
    "Insulin": "2-Hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (weight in kg/(height in m)^2)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function - Likeliness based on family history",
    "Age": "Age (years)",
    "Outcome": "Classifier - Class variable (0 or 1) - Whether the patient has diabetes or not"
}

# Dropdown menu for displaying options
option = st.selectbox(label='Select an option:', options=['Attribute Descriptions', 'Dataset Shape', 'Classes'])

# Display selected option
if option == 'Attribute Descriptions':
    st.write('## Attributes:')
    for column, description in column_descriptions.items():
        st.write(f"**{column}:** {description}")
elif option == 'Dataset Shape':
    st.write(f'Shape of dataset is: {data.shape}')
elif option == 'Classes':
    st.write('## Class Distribution')
    class_counts = data['Outcome'].value_counts()
    st.bar_chart(class_counts)

# --------------------- CLASSIFICATION ------------------------------------ #

# Split data into features and target variable
X = normalized_data.drop(columns=['Outcome'])
y = normalized_data['Outcome']

# Sidebar for model selection and hyperparameters
st.sidebar.title("Model Selection & Hyperparameters")
classifier = st.sidebar.selectbox("Select classifier:", ["Random Forest", "SVM", "Decision Tree", "Adaboost"], index=0)

# Create a session state object (so the best variables persist and don't keep resetting)
session_state = st.session_state
# Initialize variables to track the best model
if 'best_accuracy' not in session_state:
    session_state.best_accuracy = 0
if 'best_classifier_params' not in session_state:
    session_state.best_classifier_params = None
if 'best_classifier' not in session_state:
    session_state.best_classifier = None
if 'best_normalization' not in session_state:
    session_state.best_normalization = None


# -------- Get Parameters -------- #
def add_parameter_ui(classifier):
    params = dict()
    if classifier == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators:", 10, 200, 100)
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy",
                                                                            "log_loss"], index=0)
        max_depth_type = st.sidebar.selectbox(label="Max Depth:", options=["none", "int"],
                                              index=0)
        if max_depth_type == "none":
            max_depth = None
        else:
            max_depth = st.sidebar.slider("Max depth:", 1, 20, 10)
        max_features_type = st.sidebar.selectbox(label="Max Features:", options=["sqrt", "log2", "none", "int"],
                                                 index=0)
        if max_features_type == "none":
            max_features = None
        elif max_features_type == "int":
            max_features = st.sidebar.slider("Max Features:", 1, 9, 9)
        else:
            max_features = max_features_type
        # Other possible parameters that can be included: (Removed for now)
        # min_samples_split = st.sidebar.slider("Minimum Number of Samples to Split:", 2, 20, 2)
        # min_samples_leaf = st.sidebar.slider("Minimum Number of Samples to be Leaf Node:", 1, 10, 1)
        # Add the selected parameter values to the list
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_features"] = max_features
    elif classifier == "SVM":
        # Create sliders / menus for the parameters
        C = st.sidebar.slider("C", 0.01, 10.0)
        # Add selected parameter values to the list
        params["C"] = C
    elif classifier == "Decision Tree":
        # Create sliders / menus for the parameters
        splitter = st.sidebar.selectbox(label="Select Splitter", options=["best", "random"], index=0)
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy",
                                                                            "log_loss"], index=0)
        max_depth_type = st.sidebar.selectbox(label="Max Depth:", options=["none", "int"],
                                              index=0)
        if max_depth_type == "none":
            max_depth = None
        else:
            max_depth = st.sidebar.slider("Max depth:", 1, 20, 10)
        max_features_type = st.sidebar.selectbox(label="Max Features:", options=["sqrt", "log2", "none", "int"],
                                                 index=0)
        if max_features_type == "none":
            max_features = None
        elif max_features_type == "int":
            max_features = st.sidebar.slider("Max Features:", 1, 9, 9)
        else:
            max_features = max_features_type
        # Add selected parameter values to the list
        params["max_depth"] = max_depth
        params["splitter"] = splitter
        params["criterion"] = criterion
        params["max_features"] = max_features
    elif classifier == "Adaboost":
        # Get user selections of parameter values
        n_estimators = st.sidebar.slider("Number of Estimators:", 1, 100, 50)
        learning_rate = st.sidebar.slider('Learning rate', 0.01, 2.0, 1.0)
        # Add selected values to parameter array
        params["learning_rate"] = learning_rate
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(classifier)

# -------- Perform Classification -------- #
# Train-test split
# Split the original dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Now let's balance the train set to be a 50/50 split between classes
# Combine features and target variable into one DF for the training set
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes in the training set
class_0_train = train_data[train_data['Outcome'] == 0]  # Overrepresented class (majority)
class_1_train = train_data[train_data['Outcome'] == 1]  # Underrepresented class (minority)

# Sample with replacement minority class (1) in the training set to match the majority class (0)
class_1_train_upsampled = class_1_train.sample(n=class_0_train.shape[0], replace=True, random_state=123)

# Combine upsampled classes for the training set
balanced_train_data = pd.concat([class_0_train, class_1_train_upsampled])

# Separate features and target variable after balancing for the training set
X_train_balanced = balanced_train_data.drop(columns=['Outcome'])
y_train_balanced = balanced_train_data['Outcome']

# Train the classifier with the selected parameters
if classifier == "Random Forest":
    model = RandomForestClassifier(**params, random_state=1234)
elif classifier == "SVM":
    model = SVC(**params)
elif classifier == "Decision Tree":
    model = DecisionTreeClassifier(**params, random_state=1234)
elif classifier == "Adaboost":
    model = AdaBoostClassifier(**params, random_state=1234)

# Fit the model
model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Update the best model if the accuracy is higher
if accuracy > session_state.best_accuracy:
    session_state.best_accuracy = accuracy
    session_state.best_classifier_params = params
    session_state.best_classifier = classifier
    session_state.best_normalization = normalization_option

# Display classifier name and accuracy
st.write("""
## Classification
""")
st.write(f' ### {classifier} Accuracy: {accuracy:.4f}')
# Display the best model found
st.write("""
## Best Model
""")
st.write(f'So far, the model you have found with the highest accuracy is:')
st.write(f'Classifier: {session_state.best_classifier}')
st.write(f'Accuracy: {session_state.best_accuracy:.4f}')
st.write(f'Parameters: {session_state.best_classifier_params}')
st.write(f'Normalization: {session_state.best_normalization}')
