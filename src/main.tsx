import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Disable the WebView's built-in context menu globally; the app provides its
// own right-click menus where needed (React handlers still receive the event).
document.addEventListener("contextmenu", (e) => e.preventDefault());

// Note: the window starts hidden ("visible": false in tauri.conf.json) to
// avoid the white flash on startup. It is maximized + shown from the Rust
// side (on_page_load in lib.rs) - JS in a hidden WebView is throttled, so
// doing it here would be unreliable.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
