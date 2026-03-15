
# Develop an NMS algorithm function
import numpy as np

# test for dell local ubutu server,103  ip

def nms(boxes, scores, iou_threshold=0.5):
	"""
	非极大值抑制 (NMS)
	Args:
		boxes (ndarray): [N, 4]，每行为 [x1, y1, x2, y2]
		scores (ndarray): [N,]，置信度分数
		iou_threshold (float): IOU 阈值
	Returns:
		keep (list): 保留的 box 索引
	"""
	# If no boxes, return empty list
	if len(boxes) == 0:
		return []
	# Extract coordinates of boxes
	x1 = boxes[:, 0]  # x1 coordinate of top-left corner
	y1 = boxes[:, 1]  # y1 coordinate of top-left corner
	x2 = boxes[:, 2]  # x2 coordinate of bottom-right corner
	y2 = boxes[:, 3]  # y2 coordinate of bottom-right corner
	# Compute area of each box
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	# Sort scores in descending order and get their indices
	order = scores.argsort()[::-1]
	keep = []  # List to store kept box indices
	# Iterate while there are boxes left
	while order.size > 0:
		i = order[0]  # Index of box with highest score
		keep.append(i)  # Keep this box
		# Compute intersection coordinates with remaining boxes
		xx1 = np.maximum(x1[i], x1[order[1:]])  # max x1
		yy1 = np.maximum(y1[i], y1[order[1:]])  # max y1
		xx2 = np.minimum(x2[i], x2[order[1:]])  # min x2
		yy2 = np.minimum(y2[i], y2[order[1:]])  # min y2
		# Compute width and height of intersection
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		# Compute intersection area
		inter = w * h
		# Compute IoU (Intersection over Union)
		iou = inter / (areas[i] + areas[order[1:]] - inter)
		# Keep boxes with IoU less than or equal to threshold
		inds = np.where(iou <= iou_threshold)[0]
		# Update order, only keep boxes that passed IoU threshold
		order = order[inds + 1]
	return keep