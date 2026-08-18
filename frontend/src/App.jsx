import { useState } from "react";
import Analyze from "./pages/Analyze";
import Dashboard from "./pages/Dashboard";
import "./styles.css";

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [history, setHistory] = useState([]);

  const addResult = (result) => setHistory((h) => [result, ...h]);

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <span className="brand-name">PhishShield</span>
          <span className="brand-sub">SOC Console</span>
        </div>
        <nav className="nav">
          <button
            className={tab === "analyze" ? "nav-item active" : "nav-item"}
            onClick={() => setTab("analyze")}
          >
            Analyze
          </button>
          <button
            className={tab === "dashboard" ? "nav-item active" : "nav-item"}
            onClick={() => setTab("dashboard")}
          >
            Dashboard
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

      <footer className="footer">
        <span>Hybrid detection · rules + DistilBERT + LLM attribution</span>
        <span className="footer-dim">Local · inferred intelligence, analyst-verified</span>
      </footer>
    </div>
  );
}