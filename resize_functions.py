import numpy as np
import cv2

def convert_points(points, convertion_info):
    points = points.copy()
    points[:, 0] = points[:, 0] * \
        convertion_info['new_w'] / convertion_info['w']
    points[:, 1] = points[:, 1] * \
        convertion_info['new_h'] / convertion_info['h']

    points[:, 0] += convertion_info['x_shift']
    points[:, 1] += convertion_info['y_shift']

    points = np.round(points)

    return points


def inv_convert_points(points, convertion_info):
    points = points.copy()

    points[:, 0] -= convertion_info['x_shift']
    points[:, 1] -= convertion_info['y_shift']

    points[:, 0] = points[:, 0] * \
        convertion_info['w'] / convertion_info['new_w']
    points[:, 1] = points[:, 1] * \
        convertion_info['h'] / convertion_info['new_h']

    points = np.round(points)

    return points
    
    
def get_centers_simple(thresholded_pred_mask):
    contours, hierarchy = cv2.findContours(
        thresholded_pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0.0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = (cx, cy)
        centers.append(center)

    return centers


def resize_with_pad(image, dsize=(1536, 1024)):
    dw, dh = dsize
    dratio = dw / dh

    h, w, _ = image.shape
    ratio = w / h

    canvas = np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

    if ratio >= dratio:
        coef = w / dw
        new_w = dw
        new_h = round(h / coef)
        image = cv2.resize(image, (new_w, new_h))
        y_shift = (dh - new_h) // 2
        x_shift = 0
    else:
        coef = h / dh
        new_h = dh
        new_w = round(w / coef)
        image = cv2.resize(image, (new_w, new_h))
        x_shift = (dw - new_w) // 2
        y_shift = 0

    canvas[y_shift:y_shift+new_h, x_shift:x_shift+new_w] = image

    convertion_info = {
        'h': h,
        'w': w,
        'new_h': new_h,
        'new_w': new_w,
        'dh': dh,
        'dw': dw,
        'y_shift': y_shift,
        'x_shift': x_shift,
    }

    return canvas, convertion_info


def unresize_with_pad(image, convertion_info):
    y_shift = convertion_info['y_shift']
    x_shift = convertion_info['x_shift']
    new_h = convertion_info['new_h']
    new_w = convertion_info['new_w']
    h = convertion_info['h']
    w = convertion_info['w']

    image = image[y_shift:y_shift+new_h, x_shift:x_shift+new_w]
    image = cv2.resize(image, (w, h))

    return image
