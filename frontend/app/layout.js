import "./globals.css";
import { SidebarProvider } from "@/context/SidebarContext";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <SidebarProvider>
        <body>{children}</body>
      </SidebarProvider>
    </html>
  );
}
