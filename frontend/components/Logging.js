"use client";

import React, { useState } from "react";
import { useLog } from "@/context/LoggingContext";
import { useAlerts } from "@/context/AlertsContext";

const Logging = ({ isOpen, toggleLogging }) => {
  const { logs } = useLog();
  const { alerts } = useAlerts();
  const [selectedTab, setSelectedTab] = useState("Configuration");

  return (
    <>
      {/* Logging Section */}
      <div
        className={`fixed top-0 right-0 h-full bg-white shadow-lg transition-transform duration-300 z-50 overflow-y-scroll ${
          isOpen ? "translate-x-0" : "translate-x-full"
        } w-1/4`}
      >
        <button
          onClick={toggleLogging}
          className="absolute top-4 right-4 text-gray-800 text-xl focus:outline-none"
        >
          &times;
        </button>
        <div className="p-4">
          <h2 className="text-lg font-bold mb-4">Logs</h2>
          {/* Tab Buttons */}
          <div className="flex justify-around items-center mb-5 mt-2 pb-2">
            <button
              onClick={() => setSelectedTab("Configuration")}
              className={`px-4 py-2 text-sm font-medium ${
                selectedTab === "Configuration"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500"
              } hover:text-blue-500 transition duration-200`}
            >
              Configuration
            </button>

            <button
              onClick={() => setSelectedTab("Alerts")}
              className={`relative px-4 py-2 text-sm font-medium ${
                selectedTab === "Alerts"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500"
              } hover:text-blue-500 transition duration-200`}
            >
              Alerts
              {alerts.filter((alert) => alert.includes("Alert")).length > 0 && (
                <span className="absolute top-2 right-0 -mt-3 -mr-2 bg-red-500 text-white text-xs font-bold w-5 h-5 flex items-center justify-center rounded-full">
                  {alerts.filter((alert) => alert.includes("Alert")).length}
                </span>
              )}
            </button>
          </div>
          <div className="overflow-y-auto h-full">
            {/* Logs */}
            {selectedTab === "Configuration" &&
              logs.map((log) => (
                <p key={log} className="text-sm">
                  {log}
                </p>
              ))}

            {selectedTab === "Alerts" &&
              alerts.map((alert, index) => (
                <p key={index} className="text-sm">
                  <span
                    dangerouslySetInnerHTML={{
                      __html: alert.replace(
                        /\*\*(.*?)\*\*/g,
                        "<strong>$1</strong>"
                      ),
                    }}
                  />
                </p>
              ))}
          </div>
        </div>
      </div>

      {/* Toggle Icon */}
      {!isOpen && (
        <div
          className="fixed top-10 right-0 transform z-40 bg-gray-100 rounded-l-full p-2 cursor-pointer shadow-md hover:animate-bounce"
          onClick={toggleLogging}
        >
          <svg
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            viewBox="0 0 24 24"
            height="27px"
            width="27px"
          >
            <path d="M8 21h12a2 2 0 002-2v-2H10v2a2 2 0 11-4 0V5a2 2 0 10-4 0v3h4" />
            <path d="M19 17V5a2 2 0 00-2-2H4M15 8h-5M15 12h-5" />
          </svg>
        </div>
      )}
    </>
  );
};

export default Logging;
