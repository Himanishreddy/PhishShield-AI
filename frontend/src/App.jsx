import { useState } from "react";
import Analyze from "./pages/Analyze";
import Dashboard from "./pages/Dashboard";
import "./styles.css";

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [history, setHistory] = useState([]);

  const addResult = (result) => {
    setHistory((current) => [result, ...current]);
  };

  return (
    <div className="app">

      <header className="masthead">

        <div className="brand">
          <span className="brand-name">
            PhishShield
          </span>

          <span className="brand-tagline">
            Email safety check
          </span>
        </div>

        <nav className="nav">

          <button
            className={
              tab === "analyze"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setTab("analyze")}
          >
            Check an Email
          </button>

          <button
            className={
              tab === "dashboard"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setTab("dashboard")}
          >
            Your Results
          </button>

        </nav>

      </header>

      <main className="main">

        {tab === "analyze" ? (
          <Analyze onResult={addResult} />
        ) : (
          <Dashboard history={history} />
        )}

      </main>

    </div>
  );
}