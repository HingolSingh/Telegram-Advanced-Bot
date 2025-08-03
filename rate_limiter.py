import time
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self, max_requests=5, window_seconds=10):
        self.max_requests = max_requests              # एक यूज़र कितनी बार मैसेज भेज सकता है
        self.window_seconds = window_seconds          # कितने सेकंड के अंदर
        self.user_messages = defaultdict(deque)       # हर यूज़र के लिए टाइम स्टैम्प्स

    def is_allowed(self, user_id: int) -> bool:
        current_time = time.time()
        user_queue = self.user_messages[user_id]

        # पुराने मैसेज हटा दो जो window से बाहर हैं
        while user_queue and current_time - user_queue[0] > self.window_seconds:
            user_queue.popleft()

        # अगर अब भी लिमिट के अंदर हैं, तो allow करो
        if len(user_queue) < self.max_requests:
            user_queue.append(current_time)
            return True
        else:
            return False