"use client";

import React, { useState, useEffect } from "react";
import Sidebar from "../../components/Sidebar";
import Logging from "../../components/Logging";
import { Toaster } from "react-hot-toast";
import { useSidebarSettings } from "@/context/SidebarContext";

const VIDEO_FEED_KEYPOINTS_URL = "http://localhost:5000/video_feed_keypoints";
const VIDEO_FEED_KEYPOINTS_MULTI_URL = "http://localhost:5000/video_feed_keypoints_multi";

function Page() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggingOpen, setIsLoggingOpen] = useState(true);
  const [videoFeedURL, setVideoFeedURL] = useState("");

  const { detectionMode, modelType, sensitivity, isAlertEnabled } =
    useSidebarSettings();

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const toggleLogging = () => {
    setIsLoggingOpen(!isLoggingOpen);
  };

  useEffect(() => {
    if (detectionMode === "single") {
      setVideoFeedURL(VIDEO_FEED_KEYPOINTS_URL);
    } else {
      setVideoFeedURL(VIDEO_FEED_KEYPOINTS_MULTI_URL);
    }
  }, [detectionMode]);

  useEffect(() => {
    if (modelType === "ANN_150e_lr_1e-03_acc_8298") {
      // Load model
    } else {
      // Load another model
    }
  }, [modelType]);

  return (
    <div className="flex h-screen">
      <Toaster />
      {/* Sidebar */}
      <Sidebar isOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />

      {/* Main Content */}
      <div
        className={`transition-all duration-300 bg-gradient-to-br from-slate-50 to-sky-50 py-4 px-7 flex flex-col items-center justify-start ${
          isLoggingOpen ? "w-3/4" : "w-full"
        }`}
      >
        <h1 className="text-2xl font-bold mt-3 mb-5 text-center">
          Camera Feed
        </h1>
        <div
          className="relative w-full max-w-5xl rounded-lg shadow-md overflow-hidden"
          style={{ aspectRatio: "16 / 9" }} // Maintain 16:9 aspect ratio
        >
          <img
            // src={videoFeedURL}
            alt="Camera Stream"
            className="w-full h-full object-cover"
          />
        </div>
      </div>

      {/* Logging Section */}
      <Logging isOpen={isLoggingOpen} toggleLogging={toggleLogging} />
    </div>
  );
}

export default Page;
