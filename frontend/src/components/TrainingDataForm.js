import React, { useState, useEffect } from "react";

function TrainingDataForm({ data, onSubmit, loading }) {
  const [formData, setFormData] = useState([]);

  useEffect(() => {
    setFormData(data);
  }, [data]);

  const handleChange = (index, field, value) => {
    const newData = [...formData];
    newData[index] = {
      ...newData[index],
      [field]: field === "lift_type" ? value : Number(value),
    };
    setFormData(newData);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  if (!formData.length) {
    return <div>Loading template...</div>;
  }

  return (
    <form onSubmit={handleSubmit} className="training-data-form">
      <h2>Training Data</h2>
      <div className="form-grid">
        {formData.map((row, index) => (
          <div key={index} className="form-row">
            <div className="form-group">
              <label>Week</label>
              <input
                type="number"
                value={row.week}
                onChange={(e) =>
                  handleChange(index, "week", parseInt(e.target.value))
                }
                min="1"
                max="4"
              />
            </div>
            <div className="form-group">
              <label>Lift Type</label>
              <select
                value={row.lift_type}
                onChange={(e) =>
                  handleChange(index, "lift_type", e.target.value)
                }
              >
                <option value="Squat">Squat</option>
                <option value="Bench">Bench</option>
                <option value="Deadlift">Deadlift</option>
              </select>
            </div>
            <div className="form-group">
              <label>Weight Lifted</label>
              <input
                type="number"
                value={row.weight_lifted}
                onChange={(e) =>
                  handleChange(
                    index,
                    "weight_lifted",
                    parseFloat(e.target.value)
                  )
                }
                min="0"
                step="2.5"
              />
            </div>
            <div className="form-group">
              <label>Reps</label>
              <input
                type="number"
                value={row.reps}
                onChange={(e) =>
                  handleChange(index, "reps", parseInt(e.target.value))
                }
                min="0"
              />
            </div>
            <div className="form-group">
              <label>RPE</label>
              <input
                type="number"
                value={row.RPE}
                onChange={(e) =>
                  handleChange(index, "RPE", parseFloat(e.target.value))
                }
                min="0"
                max="10"
                step="0.5"
              />
            </div>
            <div className="form-group">
              <label>Day</label>
              <input
                type="number"
                value={row.day}
                onChange={(e) =>
                  handleChange(index, "day", parseInt(e.target.value))
                }
                min="1"
                max="7"
              />
            </div>
          </div>
        ))}
      </div>
      <button type="submit" disabled={loading}>
        {loading ? "Predicting..." : "Get Predictions"}
      </button>
    </form>
  );
}

export default TrainingDataForm;
