'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

const STORAGE_KEY = 'sidebarSettings';

// Default settings
const DEFAULT_SETTINGS = {
  detectionMode: "single",
  modelType: "ANN_150e_lr_1e-03_acc_8298",
  sensitivity: {
    reclining: 100,
    crossed_legs: 100,
    slouching: 100,
  },
};

// Create the context
const SidebarContext = createContext();

// Create a provider
export const SidebarProvider = ({ children }) => {
  // Initialize with default settings first
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [isAlertEnabled, setIsAlertEnabled] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  // Load from localStorage after mount
  useEffect(() => {
    const savedSettings = localStorage.getItem(STORAGE_KEY);
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
    setIsInitialized(true);
  }, []);

  // Save to localStorage whenever settings change, but only after initial load
  useEffect(() => {
    if (isInitialized) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }
  }, [settings, isInitialized]);

  // Create individual setters that update the entire settings object
  const setDetectionMode = (mode) => {
    setSettings(prev => ({ ...prev, detectionMode: mode }));
  };

  const setModelType = (model) => {
    setSettings(prev => ({ ...prev, modelType: model }));
  };

  const setSensitivity = (newSensitivity) => {
    setSettings(prev => ({ ...prev, sensitivity: newSensitivity }));
  };

  // Don't render children until client-side localStorage is checked
  if (!isInitialized) {
    return null;
  }

  return (
    <SidebarContext.Provider
      value={{
        detectionMode: settings.detectionMode,
        setDetectionMode,
        modelType: settings.modelType,
        setModelType,
        sensitivity: settings.sensitivity,
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
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebarSettings must be used within a SidebarProvider');
  }
  return context;
};