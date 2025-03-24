# Powerlifting Attempt Predictor  

This project uses machine learning to predict optimal powerlifting meet attempts based on training data. By analyzing past training sessions, the model estimates the weights an athlete should attempt for their squat, bench press, and deadlift on competition day.  

## How It Works  

### 1. Data Processing  
- Loads training logs from a CSV file.  
- Validates data to ensure completeness and correctness.  
- Extracts key performance metrics, such as weight lifted, reps, RPE (Rate of Perceived Exertion), and trends over time.  

### 2. Feature Engineering  
- Computes statistical features like mean, max, sum, standard deviation, and ratios.  
- Adds trends and previous-week performance data.  
- One-hot encodes lift types for better model performance.  

### 3. Model Training  
- Uses the **TabPFNClassifier**, a transformer-based probabilistic model, to train separate models for squat, bench, and deadlift.  
- Standardizes features using **StandardScaler** to improve generalization.  

### 4. Prediction  
- Generates predictions for three attempts in each lift.  
- Uses training data trends to make informed estimates about future performance.  

## Data Format  

The model requires training logs in a structured CSV format, including:  

| week | lift_type | weight_lifted | reps | RPE | day | squat_1 | squat_2 | squat_3 | bench_1 | ... | deadlift_3 |
|------|----------|--------------|------|-----|-----|---------|---------|---------|---------|-----|-----------|
| 1    | Squat    | 100          | 5    | 7   | 1   | 0       | 0       | 0       | 0       | ... | 0         |  
| 1    | Bench    | 80           | 5    | 7   | 1   | 0       | 0       | 0       | 0       | ... | 0         |  

## Future Enhancements  

- Improve accuracy with additional data sources.  
- Implement a web-based interface for easier data entry.  
- Experiment with deep learning models for enhanced predictions.  
