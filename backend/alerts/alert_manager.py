import time
from .telegram_service import TelegramService

class AlertManager:
    def __init__(self, threshold=600, cooldown=30):
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_alert_time = 0
        self.bad_posture_count = 0
        self.previous_posture = None
        self.detection_counts = {}
        self.detection_counts_multi = {}
        self.telegram_service = TelegramService()
        self.messaging_enabled = False
        
    def reset_state(self):
        self.last_alert_time = 0
        self.bad_posture_count = 0
        self.previous_posture = None
        
    def should_send_alert(self, predicted_label):
        """Combined approach: Cooldown + Stability."""
        current_time = time.time()

        # Check if the posture is improper
        if predicted_label != "proper":
            # Increment bad posture count if posture remains the same
            if predicted_label == self.previous_posture:
                self.bad_posture_count += 1
            else:
                self.bad_posture_count = 1

            # Check stability and cooldown
            if (self.bad_posture_count >= self.threshold and
                current_time - self.last_alert_time > self.cooldown):
                self.last_alert_time = current_time
                self.bad_posture_count = 0
                return True

        else:
            # Reset count if posture returns to normal
            self.bad_posture_count = 0

        self.previous_posture = predicted_label
        return False
            
    def send_alert(self, message, predicted_label=None, detection_count=0):
        if self.messaging_enabled:
            self.telegram_service.send_message(message, predicted_label, detection_count)
            
    def enable_messaging(self):
        self.messaging_enabled = True
        
    def disable_messaging(self):
        self.messaging_enabled = False