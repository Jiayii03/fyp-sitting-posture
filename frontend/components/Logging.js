"use client";

import React from "react";

const Logging = ({ isOpen, toggleLogging }) => {
  return (
    <>
      {/* Logging Section */}
      <div
        className={`fixed top-0 right-0 h-full bg-white shadow-lg transition-transform duration-300 z-50 ${
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
          <div className="overflow-y-auto h-full">
            <p className="text-sm">Log 1: System initialized...</p>
            <p className="text-sm">Log 2: Video stream started...</p>
            <p className="text-sm">Log 3: Keypoints detected...</p>
            {/* Add more logs dynamically */}
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
