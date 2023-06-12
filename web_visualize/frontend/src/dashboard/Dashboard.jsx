// import { useState, useEffect } from "react";
import Metrics from "./Metrics";
import Histogram from "./Histogram";

const Dashboard = () => {
  return (
    <div className="flex h-screen w-full flex-col">
      <div className="flex h-full justify-between">
        <div className="flex w-1/2 flex-col overflow-y-auto border-r-8 bg-gray-300">
          <h1 className="text-center text-5xl underline">Histogram</h1>
          <div>
            <Histogram />
          </div>
        </div>
        <div className="flex w-1/2 flex-col overflow-y-auto bg-gray-300">
          <h1 className="text-center text-5xl underline">Metrics</h1>
          <div className="h-full ">
            <div className="h-full ">
              <Metrics />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Dashboard;
