from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
# Configure CORS to allow all origins, methods, and headers
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Import the model training and prediction functions from main.py
from main import (
    create_template_csv, load_and_validate_data,
    train_models, predict_attempts
)

@app.route('/api/template', methods=['GET'])
def get_template():
    """Create and return template data"""
    try:
        # Create template if it doesn't exist
        if not os.path.exists('template.csv'):
            create_template_csv()
        
        # Read the template
        template_df = pd.read_csv('template.csv')
        response = jsonify(template_df.to_dict('records'))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        print("Received data:", data)  # Debug log

        # Validate required fields
        required_fields = ['week', 'lift_type', 'weight_lifted', 'reps', 'RPE', 'day']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        # Create a template with 4 weeks of data
        template_df = create_template_csv()
        print("Created template DataFrame:", template_df)  # Debug log
        
        # Find the matching row in template_df based on week and lift_type
        row_idx = template_df[
            (template_df['week'] == data['week']) & 
            (template_df['lift_type'] == data['lift_type'])
        ].index[0]
        
        # Update the matching row with the user's data
        for column in template_df.columns:
            if column in data:
                template_df.at[row_idx, column] = data[column]
        
        print("Updated template DataFrame:", template_df)  # Debug log
        
        # Save the updated template
        template_df.to_csv('template.csv', index=False)
        
        # Calculate predictions based on the current lift type's weight and RPE
        weight = float(data['weight_lifted'])
        rpe = float(data['RPE'])
        reps = int(data['reps'])
        lift_type = data['lift_type'].lower()
        
        # Estimate 1RM using RPE and reps
        # Formula: weight * (1 + (10 - RPE)/30) * (1 + 0.0333 * reps)
        estimated_1rm = weight * (1 + (10 - rpe)/30) * (1 + 0.0333 * reps)
        
        # Only return predictions for the current lift type
        predictions = {
            lift_type: {
                'attempt_1': int(estimated_1rm * 0.91),  # Conservative first attempt
                'attempt_2': int(estimated_1rm * 0.96),  # Moderate second attempt
                'attempt_3': int(estimated_1rm * 1.00)   # Challenging but achievable third attempt
            }
        }
        return jsonify(predictions)
            
    except Exception as e:
        print("Error occurred:", str(e))  # Debug log
        import traceback
        print("Traceback:", traceback.format_exc())  # Debug log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0') 