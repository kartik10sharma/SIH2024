import React, { useState } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import "./App.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [predictions, setPredictions] = useState({
    day: { predictedLoad: [], actualLoad: [] },
    week: { predictedLoad: [], actualLoad: [] },
    month: { predictedLoad: [], actualLoad: [] },
  });

  const [mape, setMapes] = useState({
    day: null,
    week: null,
    month: null,
  });

  const [date, setDate] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchPrediction = async (type) => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/predict/${type}?date=${date}`
      );
      console.log(`Fetched ${type} prediction:`, response.data);
      return response.data;
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError(`Error fetching ${type} prediction. ${err.message}`);
      return { predicted_load: [], actual_load: [], mape: null }; // Add mape default
    }
  };

  const handlePredict = async () => {
    setError(null);
    setLoading(true);
    if (!date) {
      setError("Please enter a valid date.");
      setLoading(false);
      return;
    }
    try {
      const dayPrediction = await fetchPrediction("day");
      const weekPrediction = await fetchPrediction("week");
      const monthPrediction = await fetchPrediction("month");

      setPredictions({
        day: {
          predictedLoad: dayPrediction.predicted_load,
          actualLoad: dayPrediction.actual_load,
        },
        week: {
          predictedLoad: weekPrediction.predicted_load,
          actualLoad: weekPrediction.actual_load,
        },
        month: {
          predictedLoad: monthPrediction.predicted_load,
          actualLoad: monthPrediction.actual_load,
        },
      });

      setMapes({
        day: dayPrediction.mape, // Store MAPE for day
        week: weekPrediction.mape, // Store MAPE for week
        month: monthPrediction.mape, // Store MAPE for month
      });
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const createChartData = (predictedLoad = [], actualLoad = []) => {
    console.log("Creating chart data:", { predictedLoad, actualLoad });
    return {
      labels: predictedLoad.length
        ? [...Array(predictedLoad.length).keys()].map((i) => `Block ${i + 1}`)
        : [],
      datasets: [
        {
          label: "Predicted Load",
          data: predictedLoad,
          borderColor: "rgba(255, 99, 132, 1)",
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          fill: true,
        },
        {
          label: "Actual Load",
          data: actualLoad,
          borderColor: "rgba(0, 25, 87, 1)",
          backgroundColor: "rgba(0, 25, 87, 0.2)",
          fill: true,
        },
      ],
    };
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "Load Predictions",
      },
      legend: {
        position: "top",
      },
      tooltip: {
        mode: "index",
        intersect: false,
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          color: "#d8d8d8",
        },
        title: {
          display: true,
          text: "Time Step",
          color: "#0a0a0a",
        },
        ticks: {
          color: "#0a0a0a",
        },
      },
      y: {
        display: true,
        grid: {
          color: "#d8d8d8",
        },
        title: {
          display: true,
          text: "Load Value",
          color: "#0a0a0a",
        },
        ticks: {
          color: "#0a0a0a",
        },
      },
    },
  };

  console.log("Predictions state:", predictions);

  const renderTable = (predictedLoad, actualLoad) => {
    const slicePredictedLoad = predictedLoad;
    const sliceActualLoad = actualLoad;

    return (
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Time Step</th>
              <th>Predicted Load</th>
              <th>Actual Load</th>
            </tr>
          </thead>
          <tbody>
            {slicePredictedLoad.map((predicted, index) => (
              <tr key={index}>
                <td className="table-data">{`Block ${index + 1}`}</td>
                <td className="table-data">{predicted}</td>
                <td className="table-data">
                  {sliceActualLoad[index] !== undefined
                    ? sliceActualLoad[index]
                    : "N/A"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="App">
      <h1 className="Heading">Power Demand Prediction Model</h1>
      <div className="navbar">
        <p>Team Coherence</p>
        <p>SIH Problem Statement ID: 1624</p>
      </div>
      <div className="prediction-section">
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          placeholder="YYYY-MM-DD"
          className="input-field"
        />
        <button className="predict-btn" onClick={handlePredict}>
          Predict Load
        </button>
      </div>
      {loading && <p className="api-call-loading">Loading predictions...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      <div>
        <h2 className="sub-heading">Load Prediction Results: </h2>

        <div style={{ width: "100%", height: "300px" }}>
          <h3 className="daily-heading">Daily Prediction</h3>
          <Line
            data={createChartData(
              predictions.day.predictedLoad,
              predictions.day.actualLoad
            )}
            options={options}
          />
          <p className="mape-error">
            Daily MAPE: <span>{mape.day?.toFixed(2)}%</span>
          </p>

          {renderTable(
            predictions.day.predictedLoad,
            predictions.day.actualLoad
          )}
          <p className="sidenote">Each block represents 15 mins</p>
        </div>

        <div style={{ width: "100%", height: "300px" }}>
          <h3 className="week-heading">Weekly Prediction</h3>
          <Line
            data={createChartData(
              predictions.week.predictedLoad,
              predictions.week.actualLoad
            )}
            options={options}
          />
          <p className="mape-error">
            Weekly MAPE: <span>{mape.week?.toFixed(2)}%</span>
          </p>

          {renderTable(
            predictions.week.predictedLoad,
            predictions.week.actualLoad
          )}
          <p className="sidenote">Each block represents 15 mins</p>
        </div>

        <div style={{ width: "100%", height: "300px" }}>
          <h3 className="month-heading">Monthly Prediction</h3>
          <Line
            data={createChartData(
              predictions.month.predictedLoad,
              predictions.month.actualLoad
            )}
            options={options}
          />
          <p className="mape-error">
            Monthly MAPE: <span>{mape.month?.toFixed(2)}%</span>
          </p>

          {renderTable(
            predictions.month.predictedLoad,
            predictions.month.actualLoad
          )}
          <p className="sidenote">Each block represents 15 mins</p>
        </div>
      </div>
    </div>
  );
}

export default App;
