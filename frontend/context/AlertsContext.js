"use client";

import React, { createContext, useState, useContext, useEffect } from "react";
import { io } from "socket.io-client";

const AlertsContext = createContext();

export const useAlerts = () => useContext(AlertsContext);

export const AlertsProvider = ({ children }) => {
  const [alerts, setAlerts] = useState([]);

  const addAlert = (alert) => {
    setAlerts((prevAlerts) => [...prevAlerts, alert]);
  };

  useEffect(() => {
    const socket = io("http://localhost:5000", {
      transports: ["websocket"],  // Force WebSocket transport
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
  
    socket.on("connect", () => {
      console.log("Socket connected");
    });
  
    socket.on("posture_alert", (data) => {
      console.log("Posture alert received: ", data);
      addAlert(`Posture changed to: ${data.posture}`);
    });
  
    socket.on("connect_error", (err) => {
      console.error("Connection error: ", err);
    });
  
    return () => socket.disconnect();
  }, []);

  return (
    <AlertsContext.Provider value={{ alerts, addAlert }}>
      {children}
    </AlertsContext.Provider>
  );
};
