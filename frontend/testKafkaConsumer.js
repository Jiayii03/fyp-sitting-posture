import { kafka_broker } from './config/hostConfig';

// consumer.js
const { Kafka } = require('kafkajs');

// Initialize Kafka consumer
const kafka = new Kafka({
  clientId: 'posture-consumer',
  brokers: [kafka_broker],
});

const consumer = kafka.consumer({ groupId: 'posture-group' });

const run = async () => {
  try {
    await consumer.connect();
    console.log("✅ Kafka consumer connected!");

    await consumer.subscribe({ topic: 'posture_events', fromBeginning: true });
    console.log("🛎️ Subscribed to 'posture_events' topic");

    await consumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        const event = JSON.parse(message.value.toString());
        console.log(`🔔 Posture alert received: ${event.posture}`);
      },
    });
  } catch (error) {
    console.error("❌ Error in consumer:", error);
  }
};

run();
