import json
from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Function to check if the user is premium
def is_user_premium(user_id):
    # Load user details from a JSON file (replace with your actual logic)
    with open('user_details.json', 'r') as f:
        user_details = json.load(f)
    
    # Check if the user_id is in the premium users list
    return user_id in user_details.get('premium_users', [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Load the uploaded file
    file = request.files['file']
    data = pd.read_excel(file)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Ensure that all specified features are present in the data after imputation
    features = ['Production volume', 'Electricity Generation',
                'Hydroelectricity consumption', 'Nuclear energy consumption',
                'final energy consumption', 'Electricity from fossil fuels',
                'Total land use']
    for feature in features:
        if feature not in data_imputed.columns:
            return "Error: Column '{}' not found in the uploaded file after imputation.".format(feature)

    # Calculate CO2 emissions using the provided formula
    data_imputed['CO2 Emissions'] = data_imputed['Electricity Generation'] * 0.45  # Assuming a carbon intensity of 0.45 metric tons CO2 per MWh

    # Select features and target variable
    X = data_imputed[features]
    y = data_imputed['CO2 Emissions']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to check if the user is premium
def is_user_premium(user_id):
    # Load user details from a JSON file (replace with your actual logic)
    with open('user_details.json', 'r') as f:
        user_details = json.load(f)
    
    # Check if the user_id is in the premium users list
    return user_id in user_details.get('premium_users', [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Load the uploaded file
    file = request.files['file']
    data = pd.read_excel(file)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Ensure that all specified features are present in the data after imputation
    features = ['Production volume', 'Electricity Generation',
                'Hydroelectricity consumption', 'Nuclear energy consumption',
                'final energy consumption', 'Electricity from fossil fuels',
                'Total land use']
    for feature in features:
        if feature not in data_imputed.columns:
            return "Error: Column '{}' not found in the uploaded file after imputation.".format(feature)

    # Calculate CO2 emissions using the provided formula
    data_imputed['CO2 Emissions'] = data_imputed['Electricity Generation'] * 0.45  # Assuming a carbon intensity of 0.45 metric tons CO2 per MWh

    # Select features and target variable
    X = data_imputed[features]
    y = data_imputed['CO2 Emissions']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate CO2 emissions on the testing set
    total_actual_emissions = sum(y_test)
    total_predicted_emissions = sum(y_pred)
    percentage_emissions = (total_predicted_emissions / total_actual_emissions) * 100

    # Determine sustainable measures based on the percentage of CO2 emissions
    sustainable_measures = []
    if percentage_emissions < 50:
        sustainable_measures = ['Increase energy efficiency', 'Invest in renewable energy sources',
                                'Implement carbon capture and storage technology', 'Promote sustainable transportation']
    elif 50 <= percentage_emissions < 75:
        sustainable_measures = ['Improve industrial processes', 'Enhance waste management practices',
                                'Invest in afforestation and reforestation', 'Encourage sustainable agriculture']
    elif 75 <= percentage_emissions < 90:
        sustainable_measures = ['Transition to low-carbon energy sources', 'Implement carbon pricing mechanisms',
                                'Adopt sustainable urban planning strategies', 'Strengthen international cooperation on climate action']
    else:
        sustainable_measures = ['Urgent action required: Reduce emissions across all sectors',
                                'Invest in innovative technologies for decarbonization',
                                'Implement strict regulations on greenhouse gas emissions',
                                'Raise awareness and mobilize public support for climate action']

    # Calculate CO2 emissions reduction after implementing sustainable measures
    reduction_percentage = 0.0  # Default value if no sustainable measures are implemented
    total_predicted_emissions_after = total_predicted_emissions  # Default value if no sustainable measures are implemented
    total_predicted_emissions_after_percentage = percentage_emissions  # Default value if no sustainable measures are implemented
    if sustainable_measures:
        # Apply some reduction percentage based on the effectiveness of the measures (for example, 10% reduction)
        reduction_percentage = 10.0
        total_predicted_emissions_after = total_predicted_emissions * (1 - reduction_percentage / 100)
        total_predicted_emissions_after_percentage = (total_predicted_emissions_after / total_actual_emissions) * 100

    # Render the result template with the calculated values
    return render_template('result.html', mae=mae, mse=mse, r2=r2, percentage_emissions=percentage_emissions,
                           sustainable_measures=sustainable_measures, reduction_percentage=reduction_percentage,
                           total_predicted_emissions_after_percentage=total_predicted_emissions_after_percentage)

@app.route('/premium')
def premium():
    return render_template('premium.html')

if __name__ == '__main__':
    app.run(debug=True)

