import "./globals.css";
import { SidebarProvider } from "@/context/SidebarContext";
import { LogProvider } from "@/context/LoggingContext";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <LogProvider>
        <SidebarProvider>
          <body>{children}</body>
        </SidebarProvider>
      </LogProvider>
    </html>
  );
}
