// app/api/kafkaConsumer/route.js
import { Kafka } from "kafkajs";

export async function GET(req) {
  const kafka = new Kafka({
    clientId: "posture-consumer",
    brokers: ["localhost:9092"],
  });

  const consumer = kafka.consumer({ groupId: "posture-group" });

  await consumer.connect();
  await consumer.subscribe({ topic: "posture_events", fromBeginning: false });

  const readableStream = new ReadableStream({
    start(controller) {
      consumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(`data: ${JSON.stringify(event)}\n\n`);
        },
      });
    },
  });

  return new Response(readableStream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
