import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load peaking block dataset
data = pd.read_csv('peaking_data.csv')

# Group data by week and lift_type to get block-level features
block_stats = data.groupby(['week', 'lift_type']).agg({
    'weight_lifted': ['mean', 'max', 'sum', 'std', 'min'],
    'reps': ['sum', 'mean', 'max'],
    'RPE': ['mean', 'max', 'std'],
    'day': ['count']
}).reset_index()

# Flatten column names
block_stats.columns = ['week', 'lift_type'] + [f'{col[0]}_{col[1]}' for col in block_stats.columns[2:]]

# Get targets for each week
targets = data.groupby(['week', 'lift_type']).agg({
    'squat_1': 'first', 'squat_2': 'first', 'squat_3': 'first',
    'bench_1': 'first', 'bench_2': 'first', 'bench_3': 'first',
    'deadlift_1': 'first', 'deadlift_2': 'first', 'deadlift_3': 'first'
}).reset_index()

# Merge block data with targets
block_data = pd.merge(block_stats, targets, on=['week', 'lift_type'])

# Add historical context
for col in ['weight_lifted_mean', 'weight_lifted_max', 'RPE_mean']:
    block_data[f'{col}_prev_week'] = block_data.groupby('lift_type')[col].shift(1)

# Add lift-specific features
block_data['weight_to_max_ratio'] = block_data['weight_lifted_mean'] / block_data.groupby('lift_type')['weight_lifted_max'].transform('max')
block_data['volume'] = block_data['weight_lifted_sum'] * block_data['reps_sum']
block_data['intensity'] = block_data['weight_lifted_mean'] / block_data['weight_lifted_max']
block_data['fatigue'] = block_data['RPE_mean'] * block_data['reps_sum']

# Fill NaN values with 0 for previous week features
block_data = block_data.fillna(0)

# One-hot encode lift_type
block_data = pd.get_dummies(block_data, columns=['lift_type'])

# Define features
features = ['week', 
           'weight_lifted_mean', 'weight_lifted_max', 'weight_lifted_sum', 'weight_lifted_std', 'weight_lifted_min',
           'reps_sum', 'reps_mean', 'reps_max',
           'RPE_mean', 'RPE_max', 'RPE_std',
           'day_count',
           'weight_lifted_mean_prev_week', 'weight_lifted_max_prev_week', 'RPE_mean_prev_week',
           'weight_to_max_ratio', 'volume', 'intensity', 'fatigue',
           'lift_type_Bench', 'lift_type_Deadlift', 'lift_type_Squat']

# Create separate models for each lift type and attempt
models = {}
scalers = {}
predictions = {}

for lift in ['squat', 'bench', 'deadlift']:
    for attempt in range(1, 4):
        # Filter data for this lift type
        lift_mask = block_data[f'lift_type_{lift.capitalize()}'] == 1
        X = block_data[lift_mask][features].copy()
        y = block_data[lift_mask][f'{lift}_{attempt}']
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        scalers[f'{lift}_{attempt}'] = scaler
        
        # Use cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            model = TabPFNClassifier(device='cpu')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, y_pred))
        
        # Train final model on all data
        model = TabPFNClassifier(device='cpu')
        model.fit(X_scaled, y_array)
        models[f'{lift}_{attempt}'] = model
        
        # Print average CV score
        print(f'{lift.capitalize()} Attempt {attempt} CV MSE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})')

# Create new block data with aggregated features
new_block = pd.DataFrame({
    'week': [5],
    'weight_lifted_mean': [185],
    'weight_lifted_max': [185],
    'weight_lifted_sum': [185],
    'weight_lifted_std': [0],
    'weight_lifted_min': [185],
    'reps_sum': [3],
    'reps_mean': [1],
    'reps_max': [1],
    'RPE_mean': [9.5],
    'RPE_max': [9.5],
    'RPE_std': [0],
    'day_count': [1],
    'weight_lifted_mean_prev_week': [180],
    'weight_lifted_max_prev_week': [180],
    'RPE_mean_prev_week': [9.0],
    'weight_to_max_ratio': [0.84],
    'volume': [555],
    'intensity': [1.0],
    'fatigue': [28.5],
    'lift_type_Bench': [0],
    'lift_type_Deadlift': [0],
    'lift_type_Squat': [1],
})[features]  # Ensure consistent feature order

# Predict meet day attempts for each lift type and attempt
for lift in ['squat', 'bench', 'deadlift']:
    print(f'\nPredicted {lift.capitalize()} Attempts:')
    for attempt in range(1, 4):
        new_block_scaled = scalers[f'{lift}_{attempt}'].transform(new_block)
        attempt_pred = models[f'{lift}_{attempt}'].predict(new_block_scaled)
        print(f'Attempt {attempt}: {attempt_pred[0]:.0f} lbs')