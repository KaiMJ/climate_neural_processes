import { useState } from "react";
import CalendarInput from "./Calendar";

function DataSettings() {
  const [split, setSplit] = useState("Train");

  function handleClick(name) {
    setSplit(name);
  }

  const getButtonStyle = (currentSplit) => {
    let className = " p-2 rounded-md text-lg cursor-pointer shadow w-full";
    if (split === currentSplit) {
      return className + " bg-blue-500 text-white";
    } else {
      return className + " hover:bg-blue-200 text-black";
    }
  };

  const titleStyle = "font-bold mx-auto text-xl underline";

  return (
    <div className="">
      <div className="flex flex-col">
        <div className="flex flex-col">
          <h1 className={titleStyle}>Data Split</h1>
          <div className="border border-b-2 w-full"></div>

          <button
            className={getButtonStyle("Train")}
            onClick={() => handleClick("Train")}
          >
            Train
          </button>
          <button
            className={getButtonStyle("Valid")}
            onClick={() => handleClick("Valid")}
          >
            Valid
          </button>
          <button
            className={getButtonStyle("Test")}
            onClick={() => handleClick("Test")}
          >
            Test
          </button>
          <button
            className={getButtonStyle("Custom")}
            onClick={() => handleClick("Custom")}
          >
            Custom
          </button>
        </div>
        <h1 className={titleStyle}>Dates</h1>
        {split === "Train" ? (
          <div className="flex justify-evenly">
            <h1>2003/4/01</h1>
            <h1>~</h1>
            <h1>2004/01/17</h1>
          </div>
        ) : split === "Valid" ? (
          <div className="flex justify-evenly">
            <h1>2004/01/18</h1>
            <h1>~</h1>
            <h1>2004/03/31</h1>
          </div>
        ) : split === "Test" ? (
          <div className="flex justify-evenly">
            <h1>2004/04/01</h1>
            <h1>~</h1>
            <h1>2005/03/28</h1>
          </div>
        ) : (
          <CalendarInput />
        )}
      </div>
    </div>
  );
}

export default DataSettings;
