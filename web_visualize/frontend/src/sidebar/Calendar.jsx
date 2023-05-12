import { useState } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

const CalendarInput = () => {
  const startDate = new Date("2003-04-02");
  const endDate = new Date("2005-03-28");
  const [currentDate, setCurrentDate] = useState(startDate);

  return (
    <div className="flex">
      <DatePicker
        className="shadow w-full text-center"
        selected={currentDate}
        onChange={(date) => setCurrentDate(date)}
        dateFormat="yyyy/MM/dd"
        minDate={startDate}
        maxDate={endDate}
      />
    </div>
  );
};

export default CalendarInput;
