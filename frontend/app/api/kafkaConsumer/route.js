import { Kafka } from "kafkajs";

// Initialize Kafka instance
const kafka = new Kafka({
  clientId: "posture-consumer",
  brokers: [process.env.NEXT_PUBLIC_KAFKA_BROKER],
});

// Create consumer for posture events
export async function GET(req) {
  const postureConsumer = kafka.consumer({ groupId: "posture-posture-group" });
  const alertConsumer = kafka.consumer({ groupId: "posture-alert-group" });

  await postureConsumer.connect();
  await alertConsumer.connect();

  await postureConsumer.subscribe({ topic: "posture_events", fromBeginning: false });
  await alertConsumer.subscribe({ topic: "alert_events", fromBeginning: false });

  const readableStream = new ReadableStream({
    start(controller) {
      // Run consumer for posture events
      postureConsumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(`data: ${JSON.stringify({ type: "posture", ...event })}\n\n`);
        },
      });

      // Run consumer for alert events
      alertConsumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(`data: ${JSON.stringify({ type: "alert", ...event })}\n\n`);
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
