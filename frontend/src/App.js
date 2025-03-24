import React, { useState, useEffect } from "react";
import "./App.css";
import TrainingDataForm from "./components/TrainingDataForm";
import Predictions from "./components/Predictions";

function App() {
  const [trainingData, setTrainingData] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load template data when component mounts
    fetchTemplate();
  }, []);

  const fetchTemplate = async () => {
    try {
      const response = await fetch("http://localhost:5001/api/template");
      const data = await response.json();
      setTrainingData(data);
    } catch (err) {
      setError("Failed to load template data");
      console.error(err);
    }
  };

  const handleSubmit = async (data) => {
    setLoading(true);
    setError(null);
    try {
      // Initialize predictions object
      const combinedPredictions = {
        squat: { attempt_1: 0, attempt_2: 0, attempt_3: 0 },
        bench: { attempt_1: 0, attempt_2: 0, attempt_3: 0 },
        deadlift: { attempt_1: 0, attempt_2: 0, attempt_3: 0 },
      };

      // Group data by lift type
      const liftTypes = ["Squat", "Bench", "Deadlift"];
      for (const liftType of liftTypes) {
        const liftData = data.filter(
          (row) => row.weight_lifted > 0 && row.lift_type === liftType
        );

        if (liftData.length > 0) {
          // Get predictions for each row of this lift type
          const promises = liftData.map(async (row) => {
            const response = await fetch("http://localhost:5001/api/predict", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(row),
            });

            if (!response.ok) {
              throw new Error("Failed to get predictions");
            }

            return response.json();
          });

          const results = await Promise.all(promises);
          const liftKey = liftType.toLowerCase();

          // Update predictions for this lift type if we got any results
          if (results.length > 0) {
            combinedPredictions[liftKey] = {
              attempt_1: Math.max(...results.map((r) => r[liftKey].attempt_1)),
              attempt_2: Math.max(...results.map((r) => r[liftKey].attempt_2)),
              attempt_3: Math.max(...results.map((r) => r[liftKey].attempt_3)),
            };
          }
        }
      }

      setPredictions(combinedPredictions);
    } catch (err) {
      setError("Failed to get predictions");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Powerlifting Meet Predictor</h1>
      </header>
      <main>
        <TrainingDataForm
          data={trainingData}
          onSubmit={handleSubmit}
          loading={loading}
        />
        {error && <div className="error">{error}</div>}
        {predictions && <Predictions predictions={predictions} />}
      </main>
    </div>
  );
}

export default App;
