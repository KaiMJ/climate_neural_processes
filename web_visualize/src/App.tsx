import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./components/Dashboard";
import axios from "axios";
import React from "react";

export default function App() {
  axios.interceptors.request.use(
    function (config) {
      // When the request starts, change the cursor to the spinning cursor
      document.body.classList.add("cursor-wait");
      return config;
    },
    function (error) {
      // If there's an error, remove the spinning cursor
      document.body.classList.remove("cursor-wait");
      return Promise.reject(error);
    }
  );

  axios.interceptors.response.use(
    function (response) {
      // Once the response is received, remove the spinning cursor
      document.body.classList.remove("cursor-wait");
      return response;
    },
    function (error) {
      // If there's an error in the response, remove the spinning cursor
      document.body.classList.remove("cursor-wait");
      return Promise.reject(error);
    }
  );
  // initalize with 2003-04-01
  const [currentDate, setCurrentDate] = useState(
    new Date("2003-04-01T00:00:00")
  );
  const [dataset, setDataset] = useState(1); // [0 - 6 months, 1 - 1 year]
  const [hour, setHour] = useState("0"); // [0, 23]
  const [level, setLevel] = useState("0"); // [0, 26]
  const [split, setSplit] = useState("train"); // ["train", "valid", "test"
  const [scaler, setScaler] = useState("max"); // ["max", "minmax", "standard"

  const [data, setData] = useState<string>(""); // [0, 26
  const [model, setModel] = useState("");

  const [metricsImages, setMetricsImages] = useState<object>({}); // [0, 26

  const [isLoading, setIsLoading] = useState(false); // [0, 26

  const handleSubmit = async (e: any) => {
    if (e) e.preventDefault();

    const formattedDate = currentDate.toLocaleDateString("en-CA", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
    setIsLoading(true);
    try {
      const response_images = await axios.post("/api/data", {
        dataset: dataset,
        date: formattedDate,
        hour: hour,
        level: level,
        split: split,
        scaler: scaler,
      });
      setIsLoading(false);

      setData(response_images.data);
      return;
    } catch (error) {
      return error;
    }
  };

  return (
    <div>
      <div className="flex bg-gray-900 text-white">
        <Sidebar
          model={model}
          setModel={setModel}
          setMetricsImages={setMetricsImages}
        />
        <div className="flex flex-col w-full h-screen">
          <Navbar
            currentDate={currentDate}
            setCurrentDate={setCurrentDate}
            dataset={dataset}
            setDataset={setDataset}
            hour={hour}
            setHour={setHour}
            level={level}
            setLevel={setLevel}
            split={split}
            setSplit={setSplit}
            scaler={scaler}
            setScaler={setScaler}
            handleSubmit={handleSubmit}
          />
          <div className="overflow-y-auto">
            <Dashboard data={data} level={level} metricsImages={metricsImages} />
            {data != "" && isLoading && <h1>Loading...</h1>}
          </div>
        </div>
      </div>
    </div>
  )
}
