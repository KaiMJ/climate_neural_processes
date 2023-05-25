import { useState, useEffect } from "react";
import MasterR2 from "/home/mkim/Nserver/climate_neural_processes/web_visualize/frontend/data/plots/MLP_R2_plot.png";
import PopupTemplate from "./PopUpTemplate";

const Metrics = () => {
  const [popupState, setPopupState] = useState("");

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
      <h1 className={titleStyle}>R^2</h1>
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
      <h1 className={titleStyle}>Raw-MAE</h1>
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
      <h1 className={titleStyle}>MAE</h1>
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
      <h1 className={titleStyle}>RMSE</h1>
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