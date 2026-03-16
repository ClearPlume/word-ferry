import time

import torch


class TDRGuard:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.last_sync = time.time()

    def sync_if_needed(self):
        if time.time() - self.last_sync > self.interval:
            torch.cuda.synchronize()
            self.last_sync = time.time()
