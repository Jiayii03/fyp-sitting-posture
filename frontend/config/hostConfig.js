const env = process.env.NEXT_PUBLIC_ENV || "development";

let frontend_domain;
let backend_domain;

if (env === "development") {
  frontend_domain = process.env.NEXT_PUBLIC_LOCAL_FRONTEND_URL || "http://localhost:3000";
  backend_domain = process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL || "http://localhost:5000";
} else if (env === "production") {
  frontend_domain = "https://your-rented-server.com"; // Replace with actual rented server domain
  backend_domain = "http://your-raspberry-pi-ip:5000"; // Replace with actual Raspberry Pi IP
}

export { frontend_domain, backend_domain };
