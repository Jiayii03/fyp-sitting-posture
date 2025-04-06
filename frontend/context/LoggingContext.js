"use client";

import React, { createContext, useState, useContext } from "react";

const LogContext = createContext();

export const useLog = () => useContext(LogContext);

export const LogProvider = ({ children }) => {
  const timestamp = new Date().toLocaleTimeString("en-GB");
  const [logs, setLogs] = useState([`${timestamp} - System initialised`]);

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString("en-GB");
    setLogs((prevLogs) => [...prevLogs, `${timestamp} - ${message}`]);
  };

  const clearLogs = () => {
    const timestamp = new Date().toLocaleTimeString("en-GB");
    setLogs([`${timestamp} - Logs cleared`]);
  };

  return (
    <LogContext.Provider value={{ logs, addLog, clearLogs }}>
      {children}
    </LogContext.Provider>
  );
};
