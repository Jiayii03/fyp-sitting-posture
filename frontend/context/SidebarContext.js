'use client';

import React, { createContext, useContext, useState } from 'react';

// Create the context
const SidebarContext = createContext();

// Create a provider
export const SidebarProvider = ({ children }) => {
  const [detectionMode, setDetectionMode] = useState("single"); // Radio button state
  const [modelType, setModelType] = useState("Model 1"); // Dropdown selection state
  const [sensitivity, setSensitivity] = useState({
    reclining: 50,
    crossed_legs: 50,
    slouching: 50,
  });
  const [isAlertEnabled, setIsAlertEnabled] = useState(true); // Checkbox state

  return (
    <SidebarContext.Provider
      value={{
        detectionMode,
        setDetectionMode,
        modelType,
        setModelType,
        sensitivity,
        setSensitivity,
        isAlertEnabled,
        setIsAlertEnabled,
      }}
    >
      {children}
    </SidebarContext.Provider>
  );
};

// Custom hook for using the context
export const useSidebarSettings = () => {
  return useContext(SidebarContext);
};
