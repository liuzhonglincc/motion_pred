import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

# from fast_reid.fast_reid_interfece import FastReIDInterface
import torch
import os
import random
import numpy

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.fuse_reid = None
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class STrack_no_motion(BaseTrack):
    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.fuse_reid = None
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track._tlwh
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class STrack_pred_model(BaseTrack):
    def __init__(self, tlwh, score, feat=None, feat_history=5, k=4):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        # 存储轨迹的历史帧box, 用于计算前面帧到后面帧的偏移量
        self.boxes = deque([tlwh.copy()], maxlen=feat_history)
        # 存储轨迹的历史offsets, 长度和训练时对应
        self.offsets = deque([], maxlen=k)
        self.offsets_1 = deque([[0, 0, 0, 0]]*k, maxlen=k)
        self.pred_box = None # center_x,center_y, w, h
        self.alpha = 0.9
        # 轨迹的offset,采样动量更新的方式更新
        self.offsets_2 = None
        self.a = 1
        self.fuse_reid = None

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # 轨迹前一帧到当前帧的偏移
        deta = [x - y for x, y in zip(self.xywh, new_track.xywh)]
        self.offsets.append(deta)
        self.offsets_1.append(deta)

        # 动量更新
        self.offsets_2 = deta.copy()
        # if self.offsets_2 is None:
        #     self.offsets_2 = deta.copy()
        # else:
        #     item = self.offsets_2.copy()
        #     self.offsets_2 = [self.a*a + (1-self.a)*b for a,b in zip(deta, item)]

        self._tlwh = new_track._tlwh
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.boxes.append(new_track._tlwh.copy())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        # 轨迹前一帧到当前帧的偏移
        deta = [x - y for x, y in zip(self.xywh, new_track.xywh)]
        self.offsets.append(deta)
        self.offsets_1.append(deta)
        # 动量更新
        if self.offsets_2 is None:
            self.offsets_2 = deta.copy()
        else:
            item = self.offsets_2.copy()
            self.offsets_2 = [self.a*a + (1-self.a)*b for a,b in zip(deta, item)]

        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.boxes.append(new_track._tlwh.copy())

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.offsets.append(deta)
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid
            
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def xywh_to_tlbr(xywh):
        """(center x, center y, width, height) to tlbr`.
        """
        ret = np.asarray(xywh).copy()
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class STrack_model_pred_motion(BaseTrack):
    def __init__(self, tlwh, score, feat=None, feat_history=5):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.fuse_reid = None
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.boxes = deque([tlwh.copy()]*feat_history, maxlen=feat_history)
        self.pred_box = None
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track._tlwh
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.boxes.append(new_track._tlwh.copy())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.boxes.append(new_track._tlwh.copy())

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = None
            # self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        self.total_iou_i = 0
        self.count_i = 0

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks

    def update_iou(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        dists = ious_dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou,   ious_dists是iou距离, 1-距离得到iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks
    
    def update_iou_stram(self, output_results, img, model):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if len(detections) > 0 :
            humans = [] # person
            marks = [] # mark
            for detection in detections:
                person_tlx = detection.tlbr[0]
                person_tly = detection.tlbr[1]
                person_brx = detection.tlbr[2]
                person_bry = detection.tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + self.args.mark_ration*person_w
                mark_tly = person_tly + self.args.mark_ration*person_h
                mark_brx = person_brx - self.args.mark_ration*person_w
                mark_bry = person_bry - self.args.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(numpy.array(humans), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            marks = torch.tensor(numpy.array(marks), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            
            if len(strack_pool) > 0:
                trajectories = []
                for strack in strack_pool:
                    trajectories.append(strack.tlbr)
                trajectories = torch.tensor(numpy.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            else:
                trajectories = humans.detach()

            trajectories_aligned, marks_aligned, humans_aligned = model(trajectories, marks, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
            if len(strack_pool) > 0:
                temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
                temporal_dist = self.args.weight_t*temporal_dist + (1-self.args.weight_t)*dists
            else:
                temporal_dist = dists
        else:
            temporal_dist = dists
            
        if self.frame_id == 1:
            dists_fused = dists
        elif self.frame_id == 2:
            dists_fused = temporal_dist
        else:
            trajectory_aligned_feature = []
            for strack in strack_pool:
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            spatial_dist = self.args.weight_s*spatial_dist + (1-self.args.weight_s)*dists
            dists_fused = self.args.weight_st*temporal_dist + (1-self.args.weight_st)*spatial_dist
            dists = dists_fused
                
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou,   ious_dists是iou距离, 1-距离得到iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks
    
    def update_iou_SORT(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > 0
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        dists = ious_dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou,   ious_dists是iou距离, 1-距离得到iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks

    def update_iou_no_motion(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack_no_motion(STrack_no_motion.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(ious_dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou,   ious_dists是iou距离, 1-距离得到iou
            self.total_iou_i += (1 - ious_dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack_no_motion(STrack_no_motion.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks
    
    def update_iou_no_motion_SORT(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > 0
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack_no_motion(STrack_no_motion.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(ious_dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou,   ious_dists是iou距离, 1-距离得到iou
            self.total_iou_i += (1 - ious_dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks

    def update_iou_pred_model(self, output_results, model_mp, device):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack_pred_model(STrack_pred_model.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # 预测strack_pool的下一帧的box位置
        # 使用 t帧的center_x, center_y, w, h, 以及t-1到t的(center_x, center_y, w, h)偏移作为x，初始帧的偏移量使用(0,0,0,0)代替
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             # 轨迹前一帧的 center_x, center_y, w, h
        #             strack_pre_frame = strack.tlwh_to_xywh(strack.boxes[-2]).tolist()
        #             # 轨迹前一帧到当前帧的偏移
        #             deta = [x - y for x, y in zip(strack_pre_frame, strack_now_frame)]
        #             strack_now_frame.extend(deta)
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # output_dir_ciou_loss_seed_8_20
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.offsets) == 0:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             means = [sum(items) / len(items) for items in zip(*strack.offsets)]
        #             strack_now_frame.extend(means.copy())
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        #  output_dir_ciou_loss_seed_8_30
        if len(strack_pool) > 0:
            # 当前目标的 center_x, center_y, w, h            Nx4
            data_x_1 = []
            # N个目标过去K帧的偏移        Nxkx4
            data_x_2 = []
            for strack in strack_pool:
                data_x_1.append(strack.xywh.copy().tolist())
                data_x_2.append(strack.offsets_1.copy())
            data_x_1 = torch.tensor(data_x_1, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNx4
            data_x_2 = torch.tensor(data_x_2, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNxkx4
            # 预测的是当前帧的目标到下一帧的偏移量, deta_pred维度：目标数, 4 (center_x, center_y, w, h)偏移
            P_offs = model_mp(data_x_1, data_x_2).squeeze(0)
            for i in range(0, P_offs.shape[0]):
                P_off = P_offs[i].detach().cpu().numpy()
                # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
                strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # output_dir_ciou_loss_seed_8_40
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             strack_now_frame.extend(strack.offsets_2)
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # if len(strack_pool) > 0:
        #     data_x = []
        #     indexs = []
        #     for index,strack in enumerate(strack_pool):
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 第一帧轨迹的预测位置使用该轨迹的第一帧表示
        #             strack.pred_box = STrack_pred_model.xywh_to_tlbr(strack.xywh)
        #         else:
        #             # 轨迹前一帧的 center_x, center_y, w, h
        #             strack_pre_frame = strack.tlwh_to_xywh(strack.boxes[-2]).tolist()
        #             # 轨迹前一帧到当前帧的偏移
        #             deta = [x - y for x, y in zip(strack_pre_frame, strack_now_frame)]
        #             strack_now_frame.extend(deta)
        #             data_x.append(strack_now_frame)
        #             indexs.append(index)
        #     if len(data_x) > 0:
        #         P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #         for i in range(0, P_offs.shape[0]):
        #             P_off = P_offs[i].detach().cpu().numpy()
        #             # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #             strack_pool[indexs[i]].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance_1(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(ious_dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - ious_dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack_no_motion(STrack_no_motion.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance_1(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks
    
    def update_iou_pred_model_stram(self, output_results, model_mp, device, model):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack_pred_model(STrack_pred_model.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # 预测strack_pool的下一帧的box位置
        # 使用 t帧的center_x, center_y, w, h, 以及t-1到t的(center_x, center_y, w, h)偏移作为x，初始帧的偏移量使用(0,0,0,0)代替
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             # 轨迹前一帧的 center_x, center_y, w, h
        #             strack_pre_frame = strack.tlwh_to_xywh(strack.boxes[-2]).tolist()
        #             # 轨迹前一帧到当前帧的偏移
        #             deta = [x - y for x, y in zip(strack_pre_frame, strack_now_frame)]
        #             strack_now_frame.extend(deta)
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # output_dir_ciou_loss_seed_8_20
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.offsets) == 0:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             means = [sum(items) / len(items) for items in zip(*strack.offsets)]
        #             strack_now_frame.extend(means.copy())
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        #  output_dir_ciou_loss_seed_8_30
        if len(strack_pool) > 0:
            # 当前目标的 center_x, center_y, w, h            Nx4
            data_x_1 = []
            # N个目标过去K帧的偏移        Nxkx4
            data_x_2 = []
            for strack in strack_pool:
                data_x_1.append(strack.xywh.copy().tolist())
                data_x_2.append(strack.offsets_1.copy())
            data_x_1 = torch.tensor(data_x_1, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNx4
            data_x_2 = torch.tensor(data_x_2, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNxkx4
            # 预测的是当前帧的目标到下一帧的偏移量, deta_pred维度：目标数, 4 (center_x, center_y, w, h)偏移
            P_offs = model_mp(data_x_1, data_x_2).squeeze(0)
            for i in range(0, P_offs.shape[0]):
                P_off = P_offs[i].detach().cpu().numpy()
                # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
                strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # output_dir_ciou_loss_seed_8_40
        # if len(strack_pool) > 0:
        #     data_x = []
        #     for strack in strack_pool:
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 初始帧, 上一帧到初始帧的偏移量使用(0,0,0,0)代替
        #             strack_now_frame.extend([0, 0, 0, 0])
        #         else:
        #             strack_now_frame.extend(strack.offsets_2)
        #         data_x.append(strack_now_frame)
        #     P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #     for i in range(0, P_offs.shape[0]):
        #         P_off = P_offs[i].detach().cpu().numpy()
        #         # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #         strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # if len(strack_pool) > 0:
        #     data_x = []
        #     indexs = []
        #     for index,strack in enumerate(strack_pool):
        #         # 当前帧的位置 center_x, center_y, w, h
        #         strack_now_frame = strack.xywh.copy().tolist()
        #         if len(strack.boxes) == 1:
        #             # 第一帧轨迹的预测位置使用该轨迹的第一帧表示
        #             strack.pred_box = STrack_pred_model.xywh_to_tlbr(strack.xywh)
        #         else:
        #             # 轨迹前一帧的 center_x, center_y, w, h
        #             strack_pre_frame = strack.tlwh_to_xywh(strack.boxes[-2]).tolist()
        #             # 轨迹前一帧到当前帧的偏移
        #             deta = [x - y for x, y in zip(strack_pre_frame, strack_now_frame)]
        #             strack_now_frame.extend(deta)
        #             data_x.append(strack_now_frame)
        #             indexs.append(index)
        #     if len(data_x) > 0:
        #         P_offs = model_mp(torch.tensor(data_x, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)).squeeze(0)
        #         for i in range(0, P_offs.shape[0]):
        #             P_off = P_offs[i].detach().cpu().numpy()
        #             # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
        #             strack_pool[indexs[i]].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # Associate with high score detection boxes
        dists = matching.iou_distance_1(strack_pool, detections)

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if len(detections) > 0 :
            humans = [] # person
            marks = [] # mark
            for detection in detections:
                person_tlx = detection.tlbr[0]
                person_tly = detection.tlbr[1]
                person_brx = detection.tlbr[2]
                person_bry = detection.tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + self.args.mark_ration*person_w
                mark_tly = person_tly + self.args.mark_ration*person_h
                mark_brx = person_brx - self.args.mark_ration*person_w
                mark_bry = person_bry - self.args.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(numpy.array(humans), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            marks = torch.tensor(numpy.array(marks), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            
            if len(strack_pool) > 0:
                trajectories = []
                for strack in strack_pool:
                    trajectories.append(strack.tlbr)
                trajectories = torch.tensor(numpy.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.args.device)
            else:
                trajectories = humans.detach()

            trajectories_aligned, marks_aligned, humans_aligned = model(trajectories, marks, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
            if len(strack_pool) > 0:
                temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
                temporal_dist = self.args.weight_t*temporal_dist + (1-self.args.weight_t)*dists
            else:
                temporal_dist = dists
        else:
            temporal_dist = dists
            
        if self.frame_id == 1:
            dists_fused = dists
        elif self.frame_id == 2:
            dists_fused = temporal_dist
        else:
            trajectory_aligned_feature = []
            for strack in strack_pool:
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            spatial_dist = self.args.weight_s*spatial_dist + (1-self.args.weight_s)*dists
            dists_fused = self.args.weight_st*temporal_dist + (1-self.args.weight_st)*spatial_dist
            dists = dists_fused
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack_no_motion(STrack_no_motion.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance_1(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks

    def update_iou_pred_model_SORT(self, output_results, model_mp, device):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > 0
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack_pred_model(STrack_pred_model.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        #  output_dir_ciou_loss_seed_8_30
        if len(strack_pool) > 0:
            # 当前目标的 center_x, center_y, w, h            Nx4
            data_x_1 = []
            # N个目标过去K帧的偏移        Nxkx4
            data_x_2 = []
            for strack in strack_pool:
                data_x_1.append(strack.xywh.copy().tolist())
                data_x_2.append(strack.offsets_1.copy())
            data_x_1 = torch.tensor(data_x_1, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNx4
            data_x_2 = torch.tensor(data_x_2, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(device)   # 1xNxkx4
            # 预测的是当前帧的目标到下一帧的偏移量, deta_pred维度：目标数, 4 (center_x, center_y, w, h)偏移
            P_offs = model_mp(data_x_1, data_x_2).squeeze(0)
            for i in range(0, P_offs.shape[0]):
                P_off = P_offs[i].detach().cpu().numpy()
                # 将每条轨迹预测的偏移加上原始位置得到预测的位置, 将center_x, center_y, w, h 转为 tlbr
                strack_pool[i].pred_box = STrack_pred_model.xywh_to_tlbr(numpy.add(strack_pool[i].xywh, P_off))

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance_1(strack_pool, detections)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(ious_dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - ious_dists[itracked][idet])
            self.count_i += 1

            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            # 统计已经匹配上的匹配对，预测位置和检测位置间的iou
            self.total_iou_i += (1 - dists[itracked][idet])
            self.count_i += 1

            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
