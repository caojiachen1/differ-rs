import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Note: the window starts hidden ("visible": false in tauri.conf.json) to
// avoid the white flash on startup. It is maximized + shown from the Rust
// side (on_page_load in lib.rs) - JS in a hidden WebView is throttled, so
// doing it here would be unreliable.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
