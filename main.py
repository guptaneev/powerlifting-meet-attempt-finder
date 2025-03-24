import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import StandardScaler
import os

# Define features globally
features = [
    'week',
    'weight_lifted_mean', 'weight_lifted_max', 'weight_lifted_sum', 'weight_lifted_std', 'weight_lifted_min',
    'reps_sum', 'reps_mean', 'reps_max',
    'RPE_mean', 'RPE_max', 'RPE_std',
    'day_count',
    'weight_to_max_ratio', 'volume', 'intensity', 'fatigue',
    'weight_lifted_mean_trend', 'RPE_mean_trend', 'volume_trend',
    'weight_lifted_mean_prev_week', 'weight_lifted_max_prev_week', 'RPE_mean_prev_week',
    'lift_type_Bench', 'lift_type_Deadlift', 'lift_type_Squat'
]

def create_template_csv():
    """Create a template CSV file for data input"""
    template_data = {
        'week': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'lift_type': ['Squat', 'Bench', 'Deadlift'] * 4,
        'weight_lifted': [0] * 12,
        'reps': [0] * 12,
        'RPE': [0] * 12,
        'day': [1] * 12,
        'squat_1': [0] * 12,
        'squat_2': [0] * 12,
        'squat_3': [0] * 12,
        'bench_1': [0] * 12,
        'bench_2': [0] * 12,
        'bench_3': [0] * 12,
        'deadlift_1': [0] * 12,
        'deadlift_2': [0] * 12,
        'deadlift_3': [0] * 12
    }
    template_df = pd.DataFrame(template_data)
    template_df.to_csv('template.csv', index=False)
    return template_df

def load_and_validate_data(file_path):
    """Load and validate the input data"""
    try:
        data = pd.read_csv(file_path)
        required_columns = [
            'week', 'lift_type', 'weight_lifted', 'reps', 'RPE', 'day',
            'squat_1', 'squat_2', 'squat_3',
            'bench_1', 'bench_2', 'bench_3',
            'deadlift_1', 'deadlift_2', 'deadlift_3'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if data is complete
        if data.isnull().any().any():
            raise ValueError("Data contains missing values. Please fill in all fields.")
        
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_models(data):
    """Train models for each lift type and attempt"""
    # For single data point case, use the data directly
    if len(data) == 1:
        block_data = pd.DataFrame({
            'week': [data['week'].iloc[0]],
            'weight_lifted_mean': [data['weight_lifted'].iloc[0]],
            'weight_lifted_max': [data['weight_lifted'].iloc[0]],
            'weight_lifted_sum': [data['weight_lifted'].iloc[0]],
            'weight_lifted_std': [0],
            'weight_lifted_min': [data['weight_lifted'].iloc[0]],
            'reps_sum': [data['reps'].iloc[0]],
            'reps_mean': [data['reps'].iloc[0]],
            'reps_max': [data['reps'].iloc[0]],
            'RPE_mean': [data['RPE'].iloc[0]],
            'RPE_max': [data['RPE'].iloc[0]],
            'RPE_std': [0],
            'day_count': [1],
            'weight_to_max_ratio': [1.0],
            'volume': [data['weight_lifted'].iloc[0] * data['reps'].iloc[0]],
            'intensity': [1.0],
            'fatigue': [data['RPE'].iloc[0] * data['reps'].iloc[0]],
            'weight_lifted_mean_trend': [0],
            'RPE_mean_trend': [0],
            'volume_trend': [0],
            'weight_lifted_mean_prev_week': [data['weight_lifted'].iloc[0]],
            'weight_lifted_max_prev_week': [data['weight_lifted'].iloc[0]],
            'RPE_mean_prev_week': [data['RPE'].iloc[0]],
            'lift_type_Bench': [1 if data['lift_type'].iloc[0] == 'Bench' else 0],
            'lift_type_Deadlift': [1 if data['lift_type'].iloc[0] == 'Deadlift' else 0],
            'lift_type_Squat': [1 if data['lift_type'].iloc[0] == 'Squat' else 0]
        })
    else:
        # Group data by week and lift_type to get weekly features
        block_stats = data.groupby(['week', 'lift_type']).agg({
            'weight_lifted': ['mean', 'max', 'sum', 'std', 'min'],
            'reps': ['sum', 'mean', 'max'],
            'RPE': ['mean', 'max', 'std'],
            'day': ['count']
        }).reset_index()

        # Flatten column names
        block_stats.columns = ['week', 'lift_type'] + [f'{col[0]}_{col[1]}' for col in block_stats.columns[2:]]

        # Get final week targets for each lift type
        targets = data[data['week'] == data['week'].max()].groupby('lift_type').agg({
            'squat_1': 'first', 'squat_2': 'first', 'squat_3': 'first',
            'bench_1': 'first', 'bench_2': 'first', 'bench_3': 'first',
            'deadlift_1': 'first', 'deadlift_2': 'first', 'deadlift_3': 'first'
        }).reset_index()

        # Merge block data with targets
        block_data = pd.merge(block_stats, targets, on='lift_type')

        # Add lift-specific features
        block_data['weight_to_max_ratio'] = block_data['weight_lifted_mean'] / block_data.groupby('lift_type')['weight_lifted_max'].transform('max')
        block_data['volume'] = block_data['weight_lifted_sum'] * block_data['reps_sum']
        block_data['intensity'] = block_data['weight_lifted_mean'] / block_data['weight_lifted_max']
        block_data['fatigue'] = block_data['RPE_mean'] * block_data['reps_sum']

        # Add trend features
        for col in ['weight_lifted_mean', 'RPE_mean', 'volume']:
            block_data[f'{col}_trend'] = block_data.groupby('lift_type')[col].diff()

        # Add historical context
        for col in ['weight_lifted_mean', 'weight_lifted_max', 'RPE_mean']:
            block_data[f'{col}_prev_week'] = block_data.groupby('lift_type')[col].shift(1)

        # Fill NaN values
        block_data = block_data.fillna(0)

        # One-hot encode lift_type
        block_data = pd.get_dummies(block_data, columns=['lift_type'])

    # Create separate models for each lift type and attempt
    models = {}
    scalers = {}

    for lift in ['squat', 'bench', 'deadlift']:
        for attempt in range(1, 4):
            # Filter data for this lift type
            lift_mask = block_data[f'lift_type_{lift.capitalize()}'] == 1
            X = block_data[lift_mask][features]
            y = block_data[lift_mask][f'{lift}_{attempt}']
            
            # Scale features while preserving feature names
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            scalers[f'{lift}_{attempt}'] = scaler
            
            # Train model on all data
            model = TabPFNClassifier(device='cpu')
            model.fit(X_scaled, y)
            models[f'{lift}_{attempt}'] = model

    return models, scalers

def predict_attempts(models, scalers):
    """Predict meet day attempts"""
    # Get the latest data point
    latest_data = pd.read_csv('template.csv').iloc[0]
    
    # For single data point case, use a simple calculation
    if len(pd.read_csv('template.csv')) == 1:
        weight = latest_data['weight_lifted']
        rpe = latest_data['RPE']
        reps = latest_data['reps']
        
        # Simple prediction based on RPE and reps
        predictions = {}
        for lift in ['squat', 'bench', 'deadlift']:
            predictions[lift] = {}
            # First attempt: 95% of current weight
            predictions[lift]['attempt_1'] = int(weight * 0.95)
            # Second attempt: Current weight
            predictions[lift]['attempt_2'] = int(weight)
            # Third attempt: 105% of current weight
            predictions[lift]['attempt_3'] = int(weight * 1.05)
        
        return predictions
    
    # For multiple data points, use the model
    new_block = pd.DataFrame({
        'week': [5],
        'weight_lifted_mean': [latest_data['weight_lifted']],
        'weight_lifted_max': [latest_data['weight_lifted']],
        'weight_lifted_sum': [latest_data['weight_lifted']],
        'weight_lifted_std': [0],
        'weight_lifted_min': [latest_data['weight_lifted']],
        'reps_sum': [latest_data['reps']],
        'reps_mean': [latest_data['reps']],
        'reps_max': [latest_data['reps']],
        'RPE_mean': [latest_data['RPE']],
        'RPE_max': [latest_data['RPE']],
        'RPE_std': [0],
        'day_count': [1],
        'weight_to_max_ratio': [1.0],
        'volume': [latest_data['weight_lifted'] * latest_data['reps']],
        'intensity': [1.0],
        'fatigue': [latest_data['RPE'] * latest_data['reps']],
        'weight_lifted_mean_trend': [0],
        'RPE_mean_trend': [0],
        'volume_trend': [0],
        'weight_lifted_mean_prev_week': [latest_data['weight_lifted']],
        'weight_lifted_max_prev_week': [latest_data['weight_lifted']],
        'RPE_mean_prev_week': [latest_data['RPE']],
        'lift_type_Bench': [1 if latest_data['lift_type'] == 'Bench' else 0],
        'lift_type_Deadlift': [1 if latest_data['lift_type'] == 'Deadlift' else 0],
        'lift_type_Squat': [1 if latest_data['lift_type'] == 'Squat' else 0],
    })[features]  # Ensure consistent feature order

    # Predict meet day attempts for each lift type and attempt
    predictions = {}
    for lift in ['squat', 'bench', 'deadlift']:
        predictions[lift] = {}
        for attempt in range(1, 4):
            new_block_scaled = pd.DataFrame(
                scalers[f'{lift}_{attempt}'].transform(new_block),
                columns=new_block.columns,
                index=new_block.index
            )
            attempt_pred = models[f'{lift}_{attempt}'].predict(new_block_scaled)
            predictions[lift][f'attempt_{attempt}'] = int(attempt_pred[0])

    return predictions

def main():
    # Check if template exists, if not create it
    if not os.path.exists('template.csv'):
        create_template_csv()
        return
    
    # Load and validate data
    data = load_and_validate_data('template.csv')
    if data is None:
        return

    # Train models and get predictions
    models, scalers = train_models(data)
    predictions = predict_attempts(models, scalers)
    
    # Print predictions
    for lift, attempts in predictions.items():
        print(f'\nPredicted {lift.capitalize()} Attempts:')
        for attempt, weight in attempts.items():
            print(f'{attempt.replace("_", " ").title()}: {weight} lbs')

if __name__ == "__main__":
    main()