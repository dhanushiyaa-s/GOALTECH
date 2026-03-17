class Tracker:
    def __init__(self):
        self.phone_start = None
        self.total_phone_usage = 0
        self.total_frames = 0
        self.attentive_frames = 0

    def update_phone(self, s_phone):
        import time
        if s_phone:
            if self.phone_start is None:
                self.phone_start = time.time()
        else:
            if self.phone_start:
                self.total_phone_usage += int(time.time()-self.phone_start)
                self.phone_start = None

    def update_attention(self, score):
        self.total_frames += 1
        if score > 70:
            self.attentive_frames += 1

    def report(self, name):
        attention_percent = (self.attentive_frames/self.total_frames)*100 if self.total_frames else 0

        return {
            "name": name,
            "phone_time": self.total_phone_usage,
            "attention": attention_percent
        }