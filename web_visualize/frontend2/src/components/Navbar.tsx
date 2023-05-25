import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

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
          <div className="flex w-1/4 justify-evenly border-x-2">
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
          </div>
          <div className="flex w-1/4 justify-evenly border-x-2">
            <h1 className="text-2xl">Level: </h1>
            {/* input for 0~26 */}
            <input
              className={selectStyle}
              type="number"
              min={0}
              max={26}
              value={level}
              onChange={(e) => {
                let value = parseInt(e.target.value);
                if (value) {
                  if (value < 0) value = 0;
                  if (value > 26) value = 26;
                  setLevel(value);
                }
              }}
            />
          </div>
          <div className="flex w-1/4 justify-evenly border-l-2">
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
          </div>
        </div>
        <div className="h-16 flex items-center border-b-2">
          <h1 className="w-72 text-2xl font-bold ml-10 text-gray-300 border-r-4">
            Current Timestep:
          </h1>
          <div className="flex justify-between w-full items-center ">
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
                type="number"
                min="0"
                max="23"
                value={hour}
                onChange={(e) => {
                  let value = parseInt(e.target.value);
                  if (value) {
                    if (value < 0) value = 0;
                    if (value > 23) value = 23;
                    setHour(value);
                  }
                }}
              ></input>
            </div>
            <div>
              {/* <button
                type="button"
                className="p-2 bg-green-500 mr-10 rounded-md"
              >
                Play
              </button>
              <button type="button" className="p-2 bg-red-500 mr-10 rounded-md">
                Stop
              </button> */}
              <button
                type="submit"
                className="p-2 bg-blue-500 mr-10 rounded-md"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
