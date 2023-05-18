import ModelSettings from "./sidebar/ModelSettings";
import DataSettings from "./sidebar/DataSettings";
import Dashboard from "./dashboard/Dashboard";
function App() {
  return (
    <div className="flex">
      <div className="w-1/6 border-2">
        <ModelSettings />
        <DataSettings />
      </div>
      <Dashboard />
    </div>
  );
}

export default App;
