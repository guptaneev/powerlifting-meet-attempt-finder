import React from "react";

function Predictions({ predictions }) {
  if (!predictions) return null;

  return (
    <div className="predictions">
      <h2>Predicted Attempts</h2>
      <div className="predictions-grid">
        {Object.entries(predictions).map(([lift, attempts]) => (
          <div key={lift} className="lift-predictions">
            <h3>{lift}</h3>
            <div className="attempts">
              {Object.entries(attempts).map(([attempt, weight]) => (
                <div key={attempt} className="attempt">
                  <span className="attempt-number">
                    {attempt
                      .replace("_", " ")
                      .replace(/\b\w/g, (l) => l.toUpperCase())}
                    :
                  </span>
                  <span className="weight">{weight} lbs</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Predictions;
