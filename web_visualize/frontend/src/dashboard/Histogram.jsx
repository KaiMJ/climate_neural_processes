import { useState, useEffect } from "react";
import MasterR2 from "/Users/kaimkim/kaimj/climate_neural_processes/notebooks/plots/MASTER_R2.jpg";
import PopupTemplate from "./PopUpTemplate";

const Metrics = () => {
  const [popupState, setPopupState] = useState("");
  const [verticalLevel, setVerticalLevel] = useState(1);

  function handleClose() {
    setPopupState("");
  }

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setPopupState("");
      }
    };

    // Add the event listener when the component mounts
    window.addEventListener("keydown", handleKeyDown);

    // Remove the event listener when the component unmounts
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const titleStyle = "text-4xl";
  return (
    <div>
      <h1 className={titleStyle}>Date : 2003/03/01</h1>
      <div
        className={
          "mx-7 flex flex-row items-center justify-between " + titleStyle
        }
      >
        <h1 className={titleStyle}>Vertical Level :</h1>
        <input
          type="number"
          min="0"
          max="26"
          className="w-20"
          value={verticalLevel}
          onChange={(e) => setVerticalLevel(e.target.value)}
        />
        <button
          className={`rounded-xl p-2 ${
            verticalLevel == 0 ? "bg-blue-500" : "bg-gray-200 hover:bg-blue-300"
          }`}
          onClick={() => setVerticalLevel(0)}
        >
          All Levels
        </button>
        <button className="rounded-xl bg-gray-200 p-2 hover:bg-blue-500">
          Confirm
        </button>
      </div>
      <h1 className={titleStyle}>Min: 0 Max: 1e10</h1>

      <h1 className={titleStyle}>Histogram</h1>
      <img
        className="w-full cursor-pointer rounded-xl p-2 hover:bg-blue-300"
        src={MasterR2}
        onClick={(e) => {
          e.preventDefault();
          setPopupState("R2Plot");
        }}
      />
      {popupState === "R2Plot" && (
        <PopupTemplate onClose={handleClose}>
          <img src={MasterR2} className="h-screen" />
          <button
            className="hover:bg-red-500 hover:text-white"
            onClick={() => setPopupState("")}
          >
            <h1 className="">Close</h1>
          </button>
        </PopupTemplate>
      )}
    </div>
  );
};

export default Metrics;
