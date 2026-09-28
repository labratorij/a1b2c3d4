import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):

    def __init__(self, dataset, num_ids_per_batch: int, num_imgs_per_id: int):
        self.num_ids_per_batch = num_ids_per_batch
        self.num_imgs_per_id = num_imgs_per_id

        self.id_to_indices = defaultdict(list)
        self.id_to_cameras = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            self.id_to_indices[row.vehicle_id].append(idx)
            self.id_to_cameras[row.vehicle_id].append(int(row.camera_id))

        self.pids = list(self.id_to_indices.keys())
        self.batch_size = num_ids_per_batch * num_imgs_per_id
        self.length = (len(self.pids) // num_ids_per_batch) * self.batch_size

    def _sample_for_id(self, pid):
        indices = self.id_to_indices[pid]
        cameras = self.id_to_cameras[pid]
        k = self.num_imgs_per_id
        if len(indices) >= k:
            by_cam = defaultdict(list)
            for i, cam in zip(indices, cameras):
                by_cam[cam].append(i)
            for v in by_cam.values():
                random.shuffle(v)
            cams = list(by_cam.keys())
            random.shuffle(cams)
            picked = []
            ci = 0
            while len(picked) < k:
                cam = cams[ci % len(cams)]
                if by_cam[cam]:
                    picked.append(by_cam[cam].pop())
                ci += 1
                if all(len(v) == 0 for v in by_cam.values()):
                    break
            while len(picked) < k:
                picked.append(random.choice(indices))
            return picked
        return list(np.random.choice(indices, size=k, replace=True))

    def __iter__(self):
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        batches = []
        for i in range(0, len(pids) - self.num_ids_per_batch + 1, self.num_ids_per_batch):
            batch_pids = pids[i:i + self.num_ids_per_batch]
            batch = []
            for pid in batch_pids:
                batch.extend(self._sample_for_id(pid))
            batches.append(batch)
        random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return self.length
