from config.settings import TELEGRAM_BOT_TOKEN, CHAT_ID, ON_RASPBERRY
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
        if predicted_label is not None and not ON_RASPBERRY:
            # Only generate and append recommendation if not on Raspberry Pi
            try:
                # Build a formatted header for the deepseek recommendation section
                deepseek_header = "*Posture Recommendation:*\n"
                
                # Generate recommendation in the background thread
                recommendation = self.recommend_service.get_recommendation(predicted_label, detection_count)
                
                # Combine the header and recommendation
                formatted_recommendation = deepseek_header + recommendation
                
                # Append the formatted recommendation block to the main message
                message += f"\n\n{formatted_recommendation}"
            except Exception as e:
                print(f"Error generating recommendation: {e}")
        
        # Example professional alert header
        professional_message = (
            "**🚨 Posture Alert 🚨**\n"
            f"**Detected Issue:** *{predicted_label}*\n\n"
            f"{message}"
        )
        
        # Check if we have valid token and chat ID before sending
        if not self.token or not self.chat_id:
            print("⚠️ Telegram alert not sent: missing bot token or chat ID")
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": professional_message, "parse_mode": "Markdown"}
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print("✅ Telegram alert sent successfully!")
            else:
                print(f"❌ Failed to send Telegram alert: {response.text}")
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")