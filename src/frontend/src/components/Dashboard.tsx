import React, { useState, useEffect } from "react";

export default function Dashboard({ data, level, metricsImages }: any) {
  const [allImages, setAllImages] = useState<string[]>([]);
  console.log(metricsImages)
  useEffect(() => {
    if (data["images"]) {
      setAllImages(data["images"]);
    }
  }, [data]);

  useEffect

  return (
    <>
      {/* Metrics */}
      {metricsImages["mae"] && <h1 className="text-5xl p-4">Metrics</h1>}
      <div className="flex overflow-x-auto w-max border-t-2 mr-40">
        {Object.entries(metricsImages).map(([key, value]: any, index: number) => {
          return (
            <div key={index} className="w-full">
              <h1 className="text-center text-5xl">{key}</h1>
              <img
                className=""
                src={`data:image/png;base64,${value}`}
                alt="MyImage"
              />
            </div>
          )
        })
        }
      </div>
      {/* Original Image */}
      {allImages.length > 0 && <h1 className="text-5xl p-4">Original Image</h1>}
      <div className="flex overflow-x-auto w-max border-t-2 mr-40">
        {allImages.map((image, i: number) => {
          return (
            <div key={i} className="w-full">
              <h1 className="text-center text-3xl">Level : {level == 0 ?
                i + 1
                : level}</h1>
              <img
                className=""
                src={`data:image/png;base64,${image}`}
                alt="MyImage"
              />
            </div>
          );
        })
        }
      </div>

    </>

  );
}
