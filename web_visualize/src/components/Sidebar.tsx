import { useState, useEffect } from "react";
import axios from "axios";


export default function Sidebar({ model, setModel, setMetricsImages }: any) {
  const [parameters, setParameters] = useState<any>({});

  function handleClick(name: string) {
    setModel(name);
  }

  useEffect(() => {
    try {
      const response = axios.post("/api/model", {
        model: model
      });
      response.then((res) => {
        setParameters(res.data);
      })

      const response_metrics = axios.post("/api/metrics", {
        model: model
      });
      response_metrics.then((res) => {
        setMetricsImages(res.data);
      })

    } catch (error) {
      console.log(error);
    }
  }, [model]);

  const getButtonStyle = (currentModel: string) => {
    const className = " p-2 rounded-md text-md cursor-pointer shadow w-full";
    if (model === currentModel) {
      return className + " bg-blue-500 text-white";
    } else {
      return className + " hover:bg-blue-200 hover:text-black";
    }
  };

  const titleStyle = "font-bold mx-auto text-lg underline ";

  return (
    <>
      <div className="w-60 flex flex-col items-start border-2 border-gray-400 ">
        <h1 className={titleStyle}>Climate Models</h1>
        <div className="border border-b-2 w-full"></div>
        <button
          className={getButtonStyle("DNN")}
          onClick={() => handleClick("DNN")}
        >
          DNN
        </button>
        <button
          className={getButtonStyle("DNN-T")}
          onClick={() => handleClick("DNN-T")}
        >
          DNN-T
        </button>
        <button
          className={getButtonStyle("Transformer")}
          onClick={() => handleClick("Transformer")}
        >
          Transformer
        </button>
        <button
          className={getButtonStyle("Transformer-T")}
          onClick={() => handleClick("Transformer-T")}
        >
          Transformer-T
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
          className={getButtonStyle("MFNP")}
          onClick={() => handleClick("MFNP")}
        >
          MFNP
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
        {Object.entries(parameters).map(([key, value]: any, index: number) => {
          return <h1 key={index} className="w-max mx-auto">
            {`${key}: `}<span className={`text-lg underline ${key == "best val R2" && "text-yellow-500"}`} >{value}</span>
          </h1>
        })}
      </div >
    </>
  );
}