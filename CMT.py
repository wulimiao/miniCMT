import cv2
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, \
	argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time

import numpy as np
import util


class CMT(object):

	DETECTOR = 'BRISK'
	DESCRIPTOR = 'BRISK'
	DESC_LENGTH = 512
	MATCHER = 'BruteForce-Hamming'
	THR_OUTLIER = 20
	THR_CONF = 0.75
	THR_RATIO = 0.8

	estimate_scale = True
	estimate_rotation = True

	def initialise(self, im_gray0, tl, br):

		# 初始化检测器、描述符、匹配器
		self.detector = cv2.FeatureDetector_create(self.DETECTOR)
		self.descriptor = cv2.DescriptorExtractor_create(self.DESCRIPTOR)
		self.matcher = cv2.DescriptorMatcher_create(self.MATCHER)

		# 在整个图像中获得初始关键点
		keypoints_cv = self.detector.detect(im_gray0)

		# 记住矩形中的关键点作为选择的关键点
		ind = util.in_rect(keypoints_cv, tl, br)
		selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
		selected_keypoints_cv, self.selected_features = self.descriptor.compute(im_gray0, selected_keypoints_cv)
		selected_keypoints = util.keypoints_cv_to_np(selected_keypoints_cv)
		num_selected_keypoints = len(selected_keypoints_cv)

		if num_selected_keypoints == 0:
			raise Exception('No keypoints found in selection')

		# 记住不在矩形中的关键点作为背景关键点
		background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
		background_keypoints_cv, background_features = self.descriptor.compute(im_gray0, background_keypoints_cv)
		_ = util.keypoints_cv_to_np(background_keypoints_cv)

		# 从1开始分配每个关键点，背景是0
		self.selected_classes = array(range(num_selected_keypoints)) + 1
		background_classes = zeros(len(background_keypoints_cv))

		# 将背景特征和选定特征叠加到数据库中
		self.features_database = vstack((background_features, self.selected_features))

		# 相同点确认
		self.database_classes = hstack((background_classes, self.selected_classes))

		# 获取方框中所选关键点之间的所有距离
		pdist = scipy.spatial.distance.pdist(selected_keypoints)
		self.squareform = scipy.spatial.distance.squareform(pdist)

		# 获取选定关键点之间的所有角度
		angles = np.empty((num_selected_keypoints, num_selected_keypoints))
		for k1, i1 in zip(selected_keypoints, range(num_selected_keypoints)):
			for k2, i2 in zip(selected_keypoints, range(num_selected_keypoints)):

				#从k1 到 k2进行比较
				v = k2 - k1

				# 计算该矢量相对于x轴的角度
				angle = math.atan2(v[1], v[0])

				# 存储角
				angles[i1, i2] = angle

		self.angles = angles

		# 找到关键点的中心
		center = np.mean(selected_keypoints, axis=0)

		# 记住矩形相对于中心的坐标
		self.center_to_tl = np.array(tl) - center
		self.center_to_tr = np.array([br[0], tl[1]]) - center
		self.center_to_br = np.array(br) - center
		self.center_to_bl = np.array([tl[0], br[1]]) - center

		# 计算每个关键点的跳数
		self.springs = selected_keypoints - center

		#设置跟踪图像
		self.im_prev = im_gray0

		# 使关键点“活跃”的关键点
		self.active_keypoints = np.copy(selected_keypoints)

		# 将类信息附加到活动的关键点
		self.active_keypoints = hstack((selected_keypoints, self.selected_classes[:, None]))

		# 记住初始关键点的个数
		self.num_initial_keypoints = len(selected_keypoints_cv)

	def estimate(self, keypoints):

		center = array((nan, nan))
		scale_estimate = nan
		med_rot = nan

		# 至少需要2个关键点
		if keypoints.size > 1:

			# 提取关键点类
			keypoint_classes = keypoints[:, 2].squeeze().astype(np.int) 

			# 保持奇异维数
			if keypoint_classes.size == 1:
				keypoint_classes = keypoint_classes[None]

			# 排序
			ind_sort = argsort(keypoint_classes)
			keypoints = keypoints[ind_sort]
			keypoint_classes = keypoint_classes[ind_sort]

			# 获取关键点的所有组合
			all_combs = array([val for val in itertools.product(range(keypoints.shape[0]), repeat=2)])	

			# 排除了与自身的比较
			all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]

			# 测量AlCOMB[0]与AlCOMBS[1]之间的距离
			ind1 = all_combs[:, 0] 
			ind2 = all_combs[:, 1]

			class_ind1 = keypoint_classes[ind1] - 1
			class_ind2 = keypoint_classes[ind2] - 1

			duplicate_classes = class_ind1 == class_ind2

			if not all(duplicate_classes):
				ind1 = ind1[~duplicate_classes]
				ind2 = ind2[~duplicate_classes]

				class_ind1 = class_ind1[~duplicate_classes]
				class_ind2 = class_ind2[~duplicate_classes]

				pts_allcombs0 = keypoints[ind1, :2]
				pts_allcombs1 = keypoints[ind2, :2]

				# 对于某些组合，这个距离可能是0。
				# 因为在一个位置上可能    有不止一个关键点。
				dists = util.L2norm(pts_allcombs0 - pts_allcombs1)

				original_dists = self.squareform[class_ind1, class_ind2]

				scalechange = dists / original_dists

				#计算角度
				angles = np.empty((pts_allcombs0.shape[0]))

				v = pts_allcombs1 - pts_allcombs0
				angles = np.arctan2(v[:, 1], v[:, 0])
				
				original_angles = self.angles[class_ind1, class_ind2]

				angle_diffs = angles - original_angles

				# 固定长角度
				long_way_angles = np.abs(angle_diffs) > math.pi

				angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(angle_diffs[long_way_angles]) * 2 * math.pi

				scale_estimate = median(scalechange)
				if not self.estimate_scale:
					scale_estimate = 1;

				med_rot = median(angle_diffs)
				if not self.estimate_rotation:
					med_rot = 0;

				keypoint_class = keypoints[:, 2].astype(np.int)
				votes = keypoints[:, :2] - scale_estimate * (util.rotate(self.springs[keypoint_class - 1], med_rot))

				# 记住包括异常值在内的所有投票
				self.votes = votes

				# 计算选票之间的成对距离
				pdist = scipy.spatial.distance.pdist(votes)

				# 计算成对距离之间的联系
				linkage = scipy.cluster.hierarchy.linkage(pdist)

				# 基于层次距离的聚类
				T = scipy.cluster.hierarchy.fcluster(linkage, self.THR_OUTLIER, criterion='distance')

				# 计算每个簇的投票数
				cnt = np.bincount(T)  # 虚拟 0 label 仍旧存在
				
				# 获得最大类
				Cmax = argmax(cnt)

				#识别内点（=最大类成员）
				inliers = T == Cmax
				# 内点 = med_dists < THR_OUTLIER

				# 记住离群点
				self.outliers = keypoints[~inliers, :]

				# 停止跟踪异常值
				keypoints = keypoints[inliers, :]

				# 去除异常选票
				votes = votes[inliers, :]

				# 计算对象中心
				center = np.mean(votes, axis=0)

		return (center, scale_estimate, med_rot, keypoints)

	def process_frame(self, im_gray):

		tracked_keypoints, _ = util.track(self.im_prev, im_gray, self.active_keypoints)
		(center, scale_estimate, rotation_estimate, tracked_keypoints) = self.estimate(tracked_keypoints)

		# 检测关键点，计算描述符
		keypoints_cv = self.detector.detect(im_gray) 
		keypoints_cv, features = self.descriptor.compute(im_gray, keypoints_cv)

		# 创建活动关键点列表
		active_keypoints = zeros((0, 3)) 

		# 为每个特征获得最好的两个匹配
		matches_all = self.matcher.knnMatch(features, self.features_database, 2)
		# 获取选定特征的所有匹配项
		if not any(isnan(center)):
			selected_matches_all = self.matcher.knnMatch(features, self.selected_features, len(self.selected_features))


		# 对于每个关键点及其描述符
		if len(keypoints_cv) > 0:
			transformed_springs = scale_estimate * util.rotate(self.springs, -rotation_estimate)
			for i in range(len(keypoints_cv)):

				# 检索关键点位置
				location = np.array(keypoints_cv[i].pt)

				# 首先：匹配整个图像
				# 计算所有描述符的距离
				matches = matches_all[i]
				distances = np.array([m.distance for m in matches])

				combined = 1 - distances / self.DESC_LENGTH

				classes = self.database_classes

				# 获得最好的和第二好的索引
				bestInd = matches[0].trainIdx
				secondBestInd = matches[1].trainIdx

				# 根据较低的计算距离比
				ratio = (1 - combined[0]) / (1 - combined[1])

				# 提取最佳匹配类
				keypoint_class = classes[bestInd]

				# 如果距离比，绝对距离满足要求，基点类不是背景
				if ratio < self.THR_RATIO and combined[0] > self.THR_CONF and keypoint_class != 0:

					# 向活动关键点中添加关键点
					new_kpt = append(location, keypoint_class)
					active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

				# 在第二步，尝试匹配困难的关键点
				# 如果结构约束成功
				if not any(isnan(center)):

					# 计算初始描述符的距离
					matches = selected_matches_all[i]				
					distances = np.array([m.distance for m in matches])
					# 基于索引的距离排序
					idxs = np.argsort(np.array([m.trainIdx for m in matches]))
					distances = distances[idxs]					

					# 把距离转换成信任区域
					confidences = 1 - distances / self.DESC_LENGTH

					# 计算与对象中心相对应的关键点位置
					relative_location = location - center

					# 计算所有跳点的距离
					displacements = util.L2norm(transformed_springs - relative_location)

					# 对于每个跳点，计算重合量。
					weight = displacements < self.THR_OUTLIER  

					combined = weight * confidences

					classes = self.selected_classes

					# 按降序排序
					sorted_conf = argsort(combined)[::-1]  #反转

					# 获得最好的和第二好的索引
					bestInd = sorted_conf[0]
					secondBestInd = sorted_conf[1]

					# 根据最低点计算距离比
					ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd])

					# 提取最佳匹配类
					keypoint_class = classes[bestInd]

					# 如果距离比，绝对距离满足要求，基点类不是背景
					if ratio < self.THR_RATIO and combined[bestInd] > self.THR_CONF and keypoint_class != 0:

						#向活动关键点添加关键点
						new_kpt = append(location, keypoint_class)

						# 检查是否已经存在同一个类
						if active_keypoints.size > 0:
							same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
							active_keypoints = np.delete(active_keypoints, same_class, axis=0)

						active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

		# 如果跟踪了一些关键点
		if tracked_keypoints.size > 0:

			# 提取关键点类
			tracked_classes = tracked_keypoints[:, 2]

			# 已经存在一些活跃的关键点
			if active_keypoints.size > 0:

				# 添加未匹配的所有跟踪关键点
				associated_classes = active_keypoints[:, 2]
				missing = ~np.in1d(tracked_classes, associated_classes)
				active_keypoints = append(active_keypoints, tracked_keypoints[missing, :], axis=0)

			# 使用所有跟踪的关键点
			else:
				active_keypoints = tracked_keypoints

		# 更新对象状态估计值 
		_ = active_keypoints
		self.center = center
		self.scale_estimate = scale_estimate
		self.rotation_estimate = rotation_estimate
		self.tracked_keypoints = tracked_keypoints
		self.active_keypoints = active_keypoints
		self.im_prev = im_gray
		self.keypoints_cv = keypoints_cv
		_ = time.time()

		self.tl = (nan, nan)
		self.tr = (nan, nan)
		self.br = (nan, nan)
		self.bl = (nan, nan)

		self.bb = array([nan, nan, nan, nan])

		self.has_result = False
		if not any(isnan(self.center)) and self.active_keypoints.shape[0] > self.num_initial_keypoints / 10:
			self.has_result = True

			tl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tl[None, :], rotation_estimate).squeeze())
			tr = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tr[None, :], rotation_estimate).squeeze())
			br = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_br[None, :], rotation_estimate).squeeze())
			bl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_bl[None, :], rotation_estimate).squeeze())

			min_x = min((tl[0], tr[0], br[0], bl[0]))
			min_y = min((tl[1], tr[1], br[1], bl[1]))
			max_x = max((tl[0], tr[0], br[0], bl[0]))
			max_y = max((tl[1], tr[1], br[1], bl[1]))

			self.tl = tl
			self.tr = tr
			self.bl = bl
			self.br = br

			self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
