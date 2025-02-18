"use client";

import React, { createContext, useState, useContext, useEffect } from "react";

const AlertsContext = createContext();

export const useAlerts = () => useContext(AlertsContext);

export const AlertsProvider = ({ children }) => {
  const [alerts, setAlerts] = useState([]);

  const addAlert = (alert) => {
    setAlerts((prevAlerts) => [...prevAlerts, alert]);
  };

  useEffect(() => {
    const eventSource = new EventSource("/api/kafkaConsumer");

    eventSource.onmessage = (event) => {
      const timestamp = new Date().toLocaleTimeString("en-GB");
      const eventData = JSON.parse(event.data);
      console.log("Kafka Event Received:", eventData);

      // Ensure 'message' exists, fallback to 'posture' if not present
      const displayMessage =
        eventData.type === "alert"
          ? `⚠️ Alert: ${eventData.message}`
          : `🪑 Posture: ${eventData.posture}`;

      addAlert(`${timestamp} - ${displayMessage}`);
    };

    eventSource.onerror = (error) => {
      console.error("Kafka EventSource Error:", error);
    };

    return () => eventSource.close();
  }, []);

  return (
    <AlertsContext.Provider value={{ alerts, addAlert }}>
      {children}
    </AlertsContext.Provider>
  );
};
