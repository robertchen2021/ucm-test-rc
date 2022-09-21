import numpy as np
import queue


class Utils:
    @staticmethod
    def movstd(x: np.array, w: int):
        res = np.zeros_like(x)
        left = int(w / 2)
        right = int(w / 2)
        if w % 2 == 0:
            left -= 1
        else:
            right += 1

        q = queue.Queue(maxsize=w)
        total_sum = 0
        total_sq_sum = 0
        num_sample = 0

        max_mov_std = -1

        for i in range(x.shape[0]):

            if q.full():
                num = q.get()
                total_sum -= num
                total_sq_sum -= num ** 2
                num_sample -= 1
            total_sum += x[i]
            total_sq_sum += x[i] ** 2
            num_sample += 1
            q.put(x[i])

            if i < left:
                continue

            mv_avg = total_sum / num_sample
            mv_var = total_sq_sum / (num_sample - 1) - mv_avg ** 2 * num_sample / (num_sample - 1)
            std = mv_var ** 0.5
            if std > max_mov_std:
                max_mov_std = std
            res[i - left] = std

        i += 1
        while i - right + 1 < x.shape[0]:
            num = q.get()
            total_sum -= num
            total_sq_sum -= num ** 2
            num_sample -= 1
            mv_avg = total_sum / num_sample
            mv_var = (total_sq_sum / num_sample - mv_avg ** 2) * (num_sample / (num_sample - 1))
            std = mv_var ** 0.5
            if std > max_mov_std:
                max_mov_std = std

            res[i - right + 1] = std
            i += 1
        return res, max_mov_std
