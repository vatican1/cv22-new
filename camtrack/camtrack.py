#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from camtrack._corners import FrameCorners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4, view_mat3x4_to_rodrigues_and_translation, eye3x4
)


def triangulate_nviews(mats, points):
    n = len(mats)
    solve_mat = np.zeros([3 * n, 4 + n])
    for i, (x, p) in enumerate(zip(points, mats)):
        solve_mat[3 * i:3 * i + 3, :4] = p
        solve_mat[3 * i:3 * i + 3, 4 + i] = -x
    A = np.linalg.svd(solve_mat)[-1][-1, :4]
    return A / A[3]


def try_add_3d_points(median_angel, left: np.ndarray, right: np.ndarray, left_view_mat: np.ndarray,
                      right_view_mat: np.ndarray,
                      intrinsic_mat: np.ndarray):
    correspondences = build_correspondences(left, right)
    print("Пытаюсь добавить 3d точки на кадрах:", left, right)
    if not correspondences:
        print("Между кадрами", left, right, "нет соответствий")
        return False, None, None
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 left_view_mat,
                                                                 right_view_mat,
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(3, 3, 1)
                                                                 # 'max_reprojection_error', 'min_triangulation_angle_deg', 'min_depth'
                                                                 )
    if len(ids) != 0 and np.arccos(median_cos) > median_angel:
        return True, new_points_3d, ids
    else:
        return False, None, None


def find_3d_2d_masks(ids_3d, ids_2d):
    mask3d = np.array([False] * len(ids_3d))
    mask2d = np.array([False] * len(ids_2d))
    for i, id_i in enumerate(ids_3d):
        for j, id_j in enumerate(ids_2d):
            if id_i == id_j:
                mask3d[i] = True
                mask2d[j] = True
    return mask3d, mask2d


def find_frames_to_add(storage_points_3d, storage_points_2d: FrameCorners, rvec_prev: np.ndarray, tvec_prev: np.ndarray,
                       intrinsic_mat: np.ndarray):
    ids_3d = storage_points_3d[1]
    ids_2d = storage_points_2d.ids.flatten()

    mask3d, mask2d = find_3d_2d_masks(ids_3d, ids_2d)
    points_3d, points_2d = storage_points_3d[0][mask3d], storage_points_2d.points[mask2d]
    retval_, r_vec_, t_vec_, inliers_ = cv2.solvePnPRansac(points_3d,
                                                           points_2d,
                                                           intrinsic_mat,
                                                           np.array([]),
                                                           reprojectionError=3,
                                                           useExtrinsicGuess=True,
                                                           rvec=rvec_prev,
                                                           tvec=tvec_prev)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        pass
        # known_view_1, known_view_2 = find_initial_frames(corner_storage, intrinsic_mat, rgb_sequence)

    # давайте сначала получим облако 3d точек
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])  # посмотрели на соответствия 2d точек
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 pose_to_view_mat3x4(known_view_1[1]),
                                                                 pose_to_view_mat3x4(known_view_2[1]),
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(20, 0, 0)
                                                                 )  # сделали триангуляцию соответствия

    storage_points_3d = [new_points_3d, ids]  # тут храним все полученные 3d точки
    print("Облако 3d точек на инициализирующих кадрах- ", len(storage_points_3d[0]), "на кадрах:", known_view_1[0],
          known_view_2[0])

    # найдём ещё какие-нибудь кадры, на которых достачно много точек и хорошие углы
    frame_number = 10
    find_frames_to_add(storage_points_3d, corner_storage[frame_number], None, None, intrinsic_mat)

    # попробуем усилить наше облако 3d точек точками из кадров между двумя, на которых инициализировался алгоритм

    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * len(corner_storage)
    assert (len(corner_storage) == len(view_mats))

    point_cloud_builder = PointCloudBuilder(storage_points_3d[1],  # id всех найденных 3d точек
                                            storage_points_3d[0])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
