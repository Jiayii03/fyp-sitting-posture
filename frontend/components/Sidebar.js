"use client";

import React, { useEffect, useRef, useState } from "react";
import { Listbox, Switch, RadioGroup } from "@headlessui/react";
import { useSidebarSettings } from "@/context/SidebarContext";
import toast from "react-hot-toast";

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const sidebarRef = useRef(null);

  const {
    detectionMode,
    setDetectionMode,
    modelType,
    setModelType,
    sensitivity,
    setSensitivity,
    isAlertEnabled,
    setIsAlertEnabled,
  } = useSidebarSettings();

  useEffect(() => {
    console.log("Settings Updated:", {
      detectionMode,
      modelType,
      sensitivity,
      isAlertEnabled,
    });

    // toast loading and success messages
    // Create a promise to simulate the loading and success of settings update
    const updatePromise = new Promise((resolve) => {
      // Simulate a delay to show the toast
      setTimeout(() => {
        resolve("Settings updated successfully!");
      }, 500); // 500ms delay for demonstration
    });

    // Show loading and success messages using toast.promise
    toast.promise(updatePromise, {
      loading: "Updating settings...",
      success: "Settings updated successfully!",
      error: "Failed to update settings!",
    });
  }, [detectionMode, modelType, sensitivity, isAlertEnabled]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
        setIsOpen(false); // Close the sidebar if clicked outside
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <>
      {/* Hover-sensitive area to open the sidebar */}
      <div
        className="fixed top-0 left-0 h-full w-2 hover:w-4 bg-transparent z-40"
        onMouseEnter={() => setIsOpen(true)}
      ></div>

      {/* Hamburger Icon */}
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed top-4 left-7 z-50 bg-white text-slate-800 py-1 px-2 rounded focus:outline-none shadow-md"
      >
        ☰
      </button>

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className={`fixed top-0 left-0 h-full bg-gradient-to-br from-slate-100 to-slate-200 text-slate-800 transform ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } transition-transform duration-300 ease-in-out w-1/5 z-50 shadow-xl rounded-r-lg`}
      >
        <button
          onClick={() => setIsOpen(false)}
          className="absolute top-4 right-4 text-slate-800 text-sm focus:outline-none"
        >
          &times;
        </button>
        <div className="p-4">
          <h2 className="text-lg font-bold mb-4">Configuration</h2>

          {/* Detection Mode */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Detection Mode</h3>
            <RadioGroup value={detectionMode} onChange={setDetectionMode}>
              <RadioGroup.Option
                value="single"
                className={({ checked }) =>
                  `${
                    checked
                      ? "bg-slate-200 hover:bg-slate-200"
                      : "hover:bg-slate-100"
                  } p-1 rounded cursor-pointer text-sm `
                }
              >
                Single Person
              </RadioGroup.Option>
              <RadioGroup.Option
                value="multi"
                className={({ checked }) =>
                  `${
                    checked
                      ? "bg-slate-200 hover:bg-slate-200"
                      : "hover:bg-slate-100"
                  } p-1 rounded cursor-pointer text-sm mt-1`
                }
              >
                Multi-Person
              </RadioGroup.Option>
            </RadioGroup>
          </div>

          {/* Model Type */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Model Type</h3>
            <Listbox value={modelType} onChange={setModelType}>
              <Listbox.Button className="w-full p-1 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-slate-500 shadow-sm transition duration-200">
                {modelType}
              </Listbox.Button>
              <Listbox.Options className="mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm transition duration-200">
                <Listbox.Option
                  key="Model 1"
                  value="Model 1"
                  className="cursor-default select-none p-1 hover:bg-slate-200 text-sm"
                >
                  Model 1
                </Listbox.Option>
                <Listbox.Option
                  key="Model 2"
                  value="Model 2"
                  className="cursor-default select-none p-1 hover:bg-slate-200 text-sm"
                >
                  Model 2
                </Listbox.Option>
                <Listbox.Option
                  key="Model 3"
                  value="Model 3"
                  className="cursor-default select-none p-1 hover:bg-slate-200 text-sm"
                >
                  Model 3
                </Listbox.Option>
              </Listbox.Options>
            </Listbox>
          </div>

          {/* Sensitivity Sliders */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Sensitivity</h3>
            {Object.keys(sensitivity).map((key) => {
              const [sliderValue, setSliderValue] = useState(sensitivity[key]);

              useEffect(() => {
                const debounceTimeout = setTimeout(() => {
                  setSensitivity((prev) => ({
                    ...prev,
                    [key]: sliderValue,
                  }));
                }, 500); // 300ms delay

                return () => clearTimeout(debounceTimeout); // Cleanup on unmount or value change
              }, [sliderValue, key, setSensitivity]);

              return (
                <div key={key} className="mb-3">
                  <label className="flex justify-between mb-1 text-xs">
                    <span className="capitalize">{key.replace("_", " ")}</span>
                    <span>{sliderValue}%</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sliderValue}
                    onChange={(e) => setSliderValue(e.target.value)}
                    className="w-full accent-slate-600"
                  />
                </div>
              );
            })}
          </div>

          {/* Alert System */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Alert System</h3>
            <Switch
              checked={isAlertEnabled}
              onChange={setIsAlertEnabled}
              className={`${
                isAlertEnabled ? "bg-slate-500" : "bg-slate-300"
              } relative inline-flex items-center h-4 rounded-full w-8`}
            >
              <span
                className={`${
                  isAlertEnabled ? "translate-x-4" : "translate-x-1"
                } inline-block w-3 h-3 transform bg-white rounded-full transition`}
              />
            </Switch>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
