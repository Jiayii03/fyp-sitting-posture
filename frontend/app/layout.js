import "./globals.css";
import { SidebarProvider } from "@/context/SidebarContext";
import { LogProvider } from "@/context/LoggingContext";
import { AlertsProvider } from "@/context/AlertsContext";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
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
