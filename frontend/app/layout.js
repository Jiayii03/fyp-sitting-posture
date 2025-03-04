import "./globals.css";
import { SidebarProvider } from "@/context/SidebarContext";
import { LogProvider } from "@/context/LoggingContext";
import { AlertsProvider } from "@/context/AlertsContext";

export const metadata = {
  title: "FYP Sitting Posture Detection",
  description: "Real-time sitting posture detection and correction system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="sitting_icon.png" />
      </head>
      <LogProvider>
        <AlertsProvider>
          <SidebarProvider>
            <body>{children}</body>
          </SidebarProvider>
        </AlertsProvider>
      </LogProvider>
    </html>
  );
}
