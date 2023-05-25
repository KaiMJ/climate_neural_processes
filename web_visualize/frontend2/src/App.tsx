import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./components/Dashboard";
import axios from "axios";

function App() {
  // initalize with 2003-04-01
  const [currentDate, setCurrentDate] = useState(
    new Date("2003-04-01T00:00:00")
  );
  const [dataset, setDataset] = useState(""); // [0 - 6 months, 1 - 1 year]
  const [hour, setHour] = useState(0); // [0, 23]
  const [level, setLevel] = useState(0); // [0, 26]
  const [split, setSplit] = useState("train"); // ["train", "valid", "test"
  const [scaler, setScaler] = useState("max"); // ["max", "minmax", "standard"

  const handleSubmit = (e: any) => {
    const formattedDate = currentDate.toLocaleDateString("en-CA", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
    console.log("dataset", dataset);
    console.log(formattedDate);
    console.log("hour", hour);
    console.log("level", level);
    console.log("split", split);
    console.log("scaler", scaler);
    console.log(axios);
    // const response = axios.post("api/data/", {
    //   dataset: dataset,
    //   date: formattedDate,
    //   hour: hour,
    //   level: level,
    //   split: split,
    //   scaler: scaler,
    // });
    // console.log(response);

    e.preventDefault();
  };

  return (
    <>
      <div className="flex bg-gray-900 text-white">
        <Sidebar />
        <div className="flex flex-col w-full">
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
          <Dashboard />
        </div>
      </div>
    </>
  );
}

export default App;
