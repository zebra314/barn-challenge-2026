import numpy as np
import logging

class SceneClassifier:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.variance_threshold = config.get('variance_threshold', 0.05)
        self.dynamic_ratio_threshold = config.get('dynamic_ratio_threshold', 0.02)
        self.min_valid_range = config.get('min_valid_range', 0.2)
        self.max_valid_range = config.get('max_valid_range', 10.0)

    def process(self, scan_history):
        data = np.array(scan_history)

        data[data == np.inf] = self.max_valid_range + 1.0
        data[data == -np.inf] = 0.0
        data = np.nan_to_num(data)

        std_devs = np.std(data, axis=0)
        mean_dists = np.mean(data, axis=0)
        valid_mask = (mean_dists > self.min_valid_range) & (mean_dists < self.max_valid_range)
        unstable_beams = (std_devs > self.variance_threshold) & valid_mask

        num_unstable = np.sum(unstable_beams)
        total_valid = np.sum(valid_mask)

        if total_valid == 0:
            ratio = 0.0
            self.logger.warning("No valid points detected in range!")
        else:
            ratio = float(num_unstable) / float(total_valid)

        is_dynamic = ratio > self.dynamic_ratio_threshold

        return {
            'is_dynamic': is_dynamic,
            'ratio': ratio,
            'unstable_count': num_unstable,
            'valid_count': total_valid
        }
