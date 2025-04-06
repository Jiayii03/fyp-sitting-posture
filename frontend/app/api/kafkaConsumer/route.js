import { Kafka, logLevel } from "kafkajs";
import { kafka_broker } from "@/config/hostConfig";

// Initialize Kafka instance
const kafka = new Kafka({
  clientId: "posture-consumer",
  brokers: [kafka_broker],
  logLevel: logLevel.NOTHING,
});

// Create consumer for posture events
export async function GET(req) {
  const postureConsumer = kafka.consumer({ groupId: "posture-posture-group" });
  const alertConsumer = kafka.consumer({ groupId: "posture-alert-group" });
  const testConsumer = kafka.consumer({ groupId: "posture-test-group" });

  // Connect consumers
  await postureConsumer.connect();
  await alertConsumer.connect();
  await testConsumer.connect();

  // In your route.js file
  try {
    await postureConsumer.subscribe({
      topic: "posture_events",
      fromBeginning: false,
    });
    await alertConsumer.subscribe({
      topic: "alert_events",
      fromBeginning: false,
    });
    await testConsumer.subscribe({
      topic: "test_events",
      fromBeginning: false,
    });
  } catch (error) {
    console.error("Error subscribing to topics:", error);
    // Return an appropriate error response
    return new Response(
      JSON.stringify({ error: "Failed to subscribe to Kafka topics" }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  const readableStream = new ReadableStream({
    start(controller) {
      // Run consumer for posture events
      postureConsumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(
            `data: ${JSON.stringify({ type: "posture", ...event })}\n\n`
          );
          console.log("Consumed posture event:", event);
        },
      });

      // Run consumer for alert events
      alertConsumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(
            `data: ${JSON.stringify({ type: "alert", ...event })}\n\n`
          );
          console.log("Consumed alert event:", event);
        },
      });

      // Run consumer for test events
      testConsumer.run({
        eachMessage: async ({ message }) => {
          const event = JSON.parse(message.value.toString());
          controller.enqueue(
            `data: ${JSON.stringify({ type: "test", ...event })}\n\n`
          );
          console.log("Consumed test event:", event);
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
