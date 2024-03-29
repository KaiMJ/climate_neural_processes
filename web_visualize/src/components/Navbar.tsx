import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { useRef } from "react";

export default function Navbar({
  currentDate,
  setCurrentDate,
  dataset,
  setDataset,
  hour,
  setHour,
  level,
  setLevel,
  split,
  setSplit,
  scaler,
  setScaler,
  handleSubmit,
}: // eslint-disable-next-line @typescript-eslint/no-explicit-any
any) {
  const handleChange = (date: Date) => {
    setCurrentDate(date);
  };
  const timeoutId = useRef<number>();

  const handleNext = () => {
    const newDate = new Date(currentDate);
    newDate.setDate(newDate.getDate() + 1);

    setCurrentDate(newDate);
    handleSubmit();
  };

  const selectStyle = "text-2xl text-black rounded-md bg-blue-200";
  return (
    <div className="text-white">
      <form onSubmit={handleSubmit}>
        <div className="h-12 flex items-center border-b-2">
          <div className="flex w-1/4 justify-evenly border-r-2">
            <h1 className="text-2xl">Dataset: </h1>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className={selectStyle + " cursor-pointer"}
            >
              <option value="0">6 Months</option>
              <option value="1">1 Year</option>
            </select>
          </div>
          {/* <div className="flex w-1/4 justify-evenly border-x-2">
            <h1 className="text-2xl">Split: </h1>
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              className={selectStyle + " cursor-pointer"}
            >
              <option value="train">Train</option>
              <option value="valid">Valid</option>
              <option value="test">Test</option>
            </select>
          </div> */}
          <div className="flex w-1/4 justify-evenly border-x-2">
            <h1 className="text-2xl">Level: </h1>
            <input
              className={selectStyle + " w-16 text-center"}
              type="number"
              value={level}
              onChange={(e) => {
                setLevel(e.target.value);
              }}
            />
          </div>
          {/* <div className="flex w-1/4 justify-evenly border-l-2">
            <h1 className="text-2xl">Scaler: </h1>
            <select
              value={scaler}
              onChange={(e) => setScaler(e.target.value)}
              className={selectStyle + " cursor-pointer"}
            >
              <option value="max">Max</option>
              <option value="minmax">MinMax</option>
              <option value="standard">Standard</option>
            </select>
          </div> */}
        </div>
        <div className="h-16 flex items-center border-b-2">
          <h1 className="w-72 text-2xl font-bold ml-10 text-gray-300 border-r-4">
            Current Timestep:
          </h1>
          <div className="flex w-full items-center">
            <div className="flex">
              <h1 className="mx-4 text-2xl font-bold">Date:</h1>

              <DatePicker
                className="w-64 text-center text-black text-2xl bg-blue-200 hover:cursor-pointer underline outline"
                selected={currentDate}
                onChange={handleChange}
                timeFormat="HH:mm:ss"
                dateFormat="yyyy-MM-dd"
              />
              <h1 className="mx-4 text-2xl font-bold">Hour:</h1>
              <input
                className="w-16 text-center text-black text-2xl bg-blue-200 outline"
                type="text"
                value={hour}
                onChange={(e) => {
                  setHour(e.target.value);
                }}
              ></input>
            </div>
            <div className="ml-96">
              <button
                type="button"
                onClick={handleNext}
                className="p-4 mx-4 bg-green-500 rounded-md"
              >
                Next
              </button>
              <button type="submit" className="p-4 bg-blue-500 rounded-md">
                Confirm
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
