"use client";

import React, { useState, useEffect, useRef } from "react";
import Sidebar from "../../components/Sidebar";
import Logging from "../../components/Logging";
import { Toaster } from "react-hot-toast";
import { useSidebarSettings } from "@/context/SidebarContext";
import { useLog } from "@/context/LoggingContext";
import { useStableEffect } from "@/util/useStableEffect";
import { Switch } from "@headlessui/react";

const VIDEO_FEED_KEYPOINTS_URL = "http://localhost:5000/video_feed_keypoints";
const VIDEO_FEED_KEYPOINTS_MULTI_URL =
  "http://localhost:5000/video_feed_keypoints_multi";

function Page() {
  const { detectionMode, modelType, sensitivity, isAlertEnabled } =
    useSidebarSettings();
  const { logs, addLog } = useLog();

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggingOpen, setIsLoggingOpen] = useState(true);
  const [videoFeedURL, setVideoFeedURL] = useState("");
  const [isCameraFeedActive, setIsCameraFeedActive] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const toggleLogging = () => {
    setIsLoggingOpen(!isLoggingOpen);
  };

  const toggleCameraFeed = () => {
    setIsCameraFeedActive((prev) => !prev);
    addLog(isCameraFeedActive ? "Camera feed stopped" : "Camera feed started");
  };

  useStableEffect(() => {
    if (detectionMode === "single") {
      setVideoFeedURL(VIDEO_FEED_KEYPOINTS_URL + "?model_type=" + modelType);
      addLog("Using single-detection mode");
    } else {
      setVideoFeedURL(
        VIDEO_FEED_KEYPOINTS_MULTI_URL + "?model_type=" + modelType
      );
      addLog("Using multi-detection mode");
    }
  }, [detectionMode]);

  useStableEffect(() => {
    addLog(`Using model: ${modelType}`);
  }, [modelType]);

  useStableEffect(() => {
    addLog(`Alerts are ${isAlertEnabled ? "enabled" : "disabled"}`);
  }, [isAlertEnabled]);

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
        <div className="flex justify-center items-center gap-x-4 mt-3 mb-7">
          <h1 className="text-2xl font-bold  text-center">Camera Feed</h1>
          <Switch
            checked={isCameraFeedActive}
            onChange={toggleCameraFeed}
            className={`${
              isCameraFeedActive ? "bg-green-500" : "bg-gray-300"
            } relative inline-flex h-6 w-11 items-center rounded-full`}
          >
            <span
              className={`${
                isCameraFeedActive ? "translate-x-6" : "translate-x-1"
              } inline-block h-4 w-4 transform rounded-full bg-white transition`}
            />
          </Switch>
        </div>
        <div
          className="relative w-full max-w-5xl rounded-lg shadow-md overflow-hidden"
          style={{ aspectRatio: "16 / 9" }} // Maintain 16:9 aspect ratio
        >
          {isCameraFeedActive ? (
            <img
              src={videoFeedURL}
              alt="Camera Stream"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-100 text-xl font-semibold">
              Camera feed is off
            </div>
          )}
        </div>
      </div>

      {/* Logging Section */}
      <Logging isOpen={isLoggingOpen} toggleLogging={toggleLogging} />
    </div>
  );
}

export default Page;
