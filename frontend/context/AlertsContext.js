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
      const eventData = JSON.parse(event.data);
      console.log("Kafka Event Received:", eventData);
      addAlert(`Posture changed to: ${eventData.posture}`);
    };

    eventSource.onerror = (error) => {
      console.error("SSE Error:", error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <AlertsContext.Provider value={{ alerts, addAlert }}>
      {children}
    </AlertsContext.Provider>
  );
};
