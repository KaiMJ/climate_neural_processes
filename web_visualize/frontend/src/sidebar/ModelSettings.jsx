import { useState } from "react";

function ModelSettings() {
  const [model, setModel] = useState("DNN");

  function handleClick(name) {
    setModel(name);
  }

  const getButtonStyle = (currentModel) => {
    let className = " p-2 rounded-md text-lg cursor-pointer shadow w-full";
    if (model === currentModel) {
      return className + " bg-blue-500 text-white";
    } else {
      return className + " hover:bg-blue-200 text-black";
    }
  };

  const titleStyle = "font-bold mx-auto text-xl underline";

  return (
    <>
      <div className="flex flex-col items-start">
        <h1 className={titleStyle}>Climate Models</h1>
        <div className="border border-b-2 w-full"></div>
        <button
          className={getButtonStyle("DNN")}
          onClick={() => handleClick("DNN")}
        >
          DNN
        </button>
        <button
          className={getButtonStyle("Transformer")}
          onClick={() => handleClick("Transformer")}
        >
          Transformer
        </button>
        <button
          className={getButtonStyle("SFNP")}
          onClick={() => handleClick("SFNP")}
        >
          SFNP
        </button>
        <button
          className={getButtonStyle("SFNP-T")}
          onClick={() => handleClick("SFNP-T")}
        >
          SFNP-T
        </button>
        <button
          className={getButtonStyle("MFNP")}
          onClick={() => handleClick("MFNP")}
        >
          MFNP
        </button>
        <button
          className={getButtonStyle("SFANP")}
          onClick={() => handleClick("SFANP")}
        >
          SFANP
        </button>
        <button
          className={getButtonStyle("SFANP-T")}
          onClick={() => handleClick("SFANP-T")}
        >
          SFANP-T
        </button>
        <button
          className={getButtonStyle("MFANP")}
          onClick={() => handleClick("MFANP")}
        >
          MFANP
        </button>
        <div className="border border-b-2 w-full"></div>
        <h1 className={titleStyle}>Model Parameters</h1>
        <div className="border border-b-2 w-full"></div>

        <h1>Layers: 256</h1>
        <div className="border border-b-2 w-full"></div>
      </div>
    </>
  );
}

export default ModelSettings;
