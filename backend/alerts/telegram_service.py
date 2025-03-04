from config.settings import TELEGRAM_BOT_TOKEN, CHAT_ID
from alerts.recommend_service import RecommendService
import threading
import requests

class TelegramService:
    def __init__(self, token=TELEGRAM_BOT_TOKEN, chat_id=CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.recommend_service = RecommendService()
        
    def send_message(self, message, predicted_label=None, detection_count=0):
        thread = threading.Thread(
            target=self._send_message_worker,
            args=(message, predicted_label, detection_count)
        )
        thread.daemon = True
        thread.start()
        
    def _send_message_worker(self, message, predicted_label, detection_count):
        """Worker function that performs the blocking deepseek call and sends the alert."""
        if predicted_label is not None:
            # Build a formatted header for the deepseek recommendation section
            deepseek_header = (
                "*Posture Recommendation:*\n"
            )
            # Generate recommendation in the background thread
            recommendation = self.recommend_service.get_recommendation(predicted_label, detection_count)
            # Combine the header and recommendation
            formatted_recommendation = deepseek_header + recommendation
            # Append the formatted recommendation block to the main message
            message += f"\n\n{formatted_recommendation}"
        
        # Example professional alert header (if you want to include it in the message)
        professional_message = (
            "**🚨 Posture Alert 🚨**\n"
            f"**Detected Issue:** *{predicted_label}*\n\n"
            f"{message}"
        )
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": professional_message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Telegram alert sent successfully!")
        else:
            print(f"❌ Failed to send Telegram alert: {response.text}")