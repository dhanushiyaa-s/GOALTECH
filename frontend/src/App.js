import React, { useEffect, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000");

function App() {
  const [report, setReport] = useState(null);
  const [finalReport, setFinalReport] = useState(null);

  useEffect(() => {
    socket.on("connect", () => {
      console.log("Connected to backend");
    });

    socket.on("report", (data) => {
      console.log("Received:", data);

      if (data.type === "final") {
        setFinalReport(data);   // ✅ final report
      } else {
        setReport(data);        // ✅ live updates
      }
    });

    socket.on("disconnect", () => {
      console.log("Disconnected from backend");
    });

    return () => {
      socket.off("report");
    };
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>Video Analysis Report</h1>

      {!report && !finalReport && <p>Analyzing video...</p>}

      {/* 🔴 LIVE DATA */}
      {report && (
        <div>
          <h2>Live Monitoring</h2>
          <p><b>Student:</b> {report.student_name}</p>
          <p><b>Phone Usage:</b> {report.phone_usage}</p>
          <p><b>Attention:</b> {report.attention_status}</p>
          <p><b>Talking:</b> {report.is_talking ? "Yes" : "No"}</p>
          <p><b>Head Direction:</b> {report.head_direction}</p>
        </div>
      )}

      {/* 🟢 FINAL REPORT */}
      {finalReport && (
        <div style={{ marginTop: "30px", borderTop: "2px solid black" }}>
          <h2>Final Report ✅</h2>
          <p><b>Student:</b> {finalReport.student_name}</p>
          <p><b>Total Phone Usage:</b> {finalReport.phone_usage} sec</p>
          <p><b>Final Attention:</b> {finalReport.attention.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default App;