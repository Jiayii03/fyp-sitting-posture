const ON_RASPBERRY = process.env.NEXT_PUBLIC_ON_RASPBERRY || "false"; // Default to false if not set
const LAPTOP_IP = process.env.NEXT_PUBLIC_LAPTOP_IP
const RASPBERRY_IP = process.env.NEXT_PUBLIC_RASPBERRY_IP

let frontend_domain;
let backend_domain;
let kafka_broker;

frontend_domain = `http://${LAPTOP_IP}:3000`;
kafka_broker = `${LAPTOP_IP}:9092`;

if (ON_RASPBERRY === "false") {
  backend_domain = `http://${LAPTOP_IP}:5000`;
} else if (ON_RASPBERRY === "true") {
  backend_domain = `http://${RASPBERRY_IP}:5001`;
}

export { frontend_domain, backend_domain, kafka_broker };
