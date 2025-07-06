import React from "react";
import { DataProcessor } from "./b";
import { UserService } from "./c";

/**
 * Main application component that orchestrates data processing and user management
 */
export function App() {
  const processor = new DataProcessor();
  const userService = new UserService();

  /**
   * Handles the main action button click
   */
  const handleAction = () => {
    const data = processor.processData([1, 2, 3, 4, 5]);
    const user = userService.getCurrentUser();
    console.log("Processed data:", data);
    console.log("Current user:", user);
  };

  return (
    <div>
      <h1>Sample App</h1>
      <button onClick={handleAction}>Process Data</button>
    </div>
  );
}

/**
 * Main entry point of the application
 */
export function main() {
  return <App />;
}
