"use client";

import React, { useRef, useState, useEffect } from "react";
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

  // Handle settings updates with debounced toast
  const [toastDebounce, setToastDebounce] = useState(null);

  const showUpdateToast = () => {
    if (toastDebounce) {
      clearTimeout(toastDebounce);
    }

    const newTimeout = setTimeout(() => {
      toast.success("Settings updated successfully!");
    }, 500);

    setToastDebounce(newTimeout);
  };

  // Cleanup toast debounce on unmount
  useEffect(() => {
    return () => {
      if (toastDebounce) {
        clearTimeout(toastDebounce);
      }
    };
  }, [toastDebounce]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  // Handle sensitivity changes with local state
  const handleSensitivityChange = (key, value) => {
    setSensitivity({
      ...sensitivity,
      [key]: parseInt(value),
    });
    showUpdateToast();
  };

  return (
    <>
      <div
        className="fixed top-0 left-0 h-full w-2 hover:w-4 bg-transparent z-40"
        onMouseEnter={() => setIsOpen(true)}
      ></div>

      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed top-4 left-7 z-50 bg-white text-slate-800 py-1 px-2 rounded focus:outline-none shadow-md"
      >
        <svg
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          viewBox="0 0 24 24"
          height="20px"
          width="20px"
        >
          <path d="M20 7h-9M14 17H5" />
          <path d="M20 17 A3 3 0 0 1 17 20 A3 3 0 0 1 14 17 A3 3 0 0 1 20 17 z" />
          <path d="M10 7 A3 3 0 0 1 7 10 A3 3 0 0 1 4 7 A3 3 0 0 1 10 7 z" />
        </svg>
      </button>

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

          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Detection Mode</h3>
            <RadioGroup
              value={detectionMode}
              onChange={(value) => {
                setDetectionMode(value);
                showUpdateToast();
              }}
            >
              <RadioGroup.Option
                value="single"
                className={({ checked }) =>
                  `${
                    checked
                      ? "bg-slate-200 hover:bg-slate-200"
                      : "hover:bg-slate-100"
                  } p-1 rounded cursor-pointer text-sm`
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

          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Model Type</h3>
            <Listbox
              value={modelType}
              onChange={(value) => {
                setModelType(value);
                showUpdateToast();
              }}
            >
              <Listbox.Button className="w-full p-1 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-slate-500 shadow-sm transition duration-200">
                {modelType}
              </Listbox.Button>
              <Listbox.Options className="mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm transition duration-200">
                {["Model 1", "Model 2", "Model 3"].map((model) => (
                  <Listbox.Option
                    key={model}
                    value={model}
                    className="cursor-default select-none p-1 hover:bg-slate-200 text-sm"
                  >
                    {model}
                  </Listbox.Option>
                ))}
              </Listbox.Options>
            </Listbox>
          </div>

          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Sensitivity</h3>
            {Object.entries(sensitivity).map(([key, value]) => (
              <div key={key} className="mb-3">
                <label className="flex justify-between mb-1 text-xs">
                  <span className="capitalize">{key.replace("_", " ")}</span>
                  <span>{value}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={value}
                  onChange={(e) => handleSensitivityChange(key, e.target.value)}
                  className="w-full accent-slate-600"
                />
              </div>
            ))}
          </div>

          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Alert System</h3>
            <Switch
              checked={isAlertEnabled}
              onChange={(checked) => {
                setIsAlertEnabled(checked);
                showUpdateToast();
              }}
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
