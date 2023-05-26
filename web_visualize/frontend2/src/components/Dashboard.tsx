import React, { useState, useEffect } from "react";

export default function Dashboard({ data, level }: any) {
  const [allImages, setAllImages] = useState<string[]>([]);

  useEffect(() => {
    if (data["images"]) {
      setAllImages(data["images"]);
    }
  }, [data]);

  return (
    <div className="flex flex-wrap w-full">
      {allImages.map((image, i: number) => {
        return (
          <div key={i} className="w-1/4">
            <h1 className="text-center w-full">Level : {level}</h1>
            <img
              className=""
              src={`data:image/png;base64,${image}`}
              alt="MyImage"
            />
          </div>
        );
      })}
    </div>
  );
}
