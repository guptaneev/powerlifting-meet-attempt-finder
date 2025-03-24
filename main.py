import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load peaking block dataset
data = pd.read_csv('peaking_data.csv')

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

# Define features
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

# Create new block data with aggregated features for week 5
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
    'weight_to_max_ratio': [0.84],
    'volume': [555],
    'intensity': [1.0],
    'fatigue': [28.5],
    'weight_lifted_mean_trend': [5],
    'RPE_mean_trend': [0.5],
    'volume_trend': [-100],
    'weight_lifted_mean_prev_week': [180],
    'weight_lifted_max_prev_week': [180],
    'RPE_mean_prev_week': [9.0],
    'lift_type_Bench': [0],
    'lift_type_Deadlift': [0],
    'lift_type_Squat': [1],
})[features]  # Ensure consistent feature order

# Predict meet day attempts for each lift type and attempt
for lift in ['squat', 'bench', 'deadlift']:
    print(f'\nPredicted {lift.capitalize()} Attempts:')
    for attempt in range(1, 4):
        new_block_scaled = pd.DataFrame(
            scalers[f'{lift}_{attempt}'].transform(new_block),
            columns=new_block.columns,
            index=new_block.index
        )
        attempt_pred = models[f'{lift}_{attempt}'].predict(new_block_scaled)
        print(f'Attempt {attempt}: {attempt_pred[0]:.0f} lbs')