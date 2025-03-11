import cv2
import numpy as np
import argparse


def order_points(pts):
    """Sorting the points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum the coordinates to find the top-left and bottom-right corners
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Subtract the coordinates to find the top-right and bottom-left corners
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def transform(image):
    try:
        # Convert the image to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the image to binary
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # Find the contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return the original image
        if not contours:
            return None

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to 4 points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # If there aren't exactly 4 points, use the bounding box instead
        if len(approx) != 4:
            print("Couldn't find exactly 4 points, using bounding box.")
            x, y, w, h = cv2.boundingRect(largest_contour)
            approx = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype="float32")

        # Order the points
        src_points = order_points(approx.reshape(-1, 2))

        # Calculate the destination width and height while maintaining aspect ratio
        max_width = int(max(np.linalg.norm(src_points[0] - src_points[1]),
                            np.linalg.norm(src_points[2] - src_points[3])))
        max_height = int(max(np.linalg.norm(src_points[0] - src_points[3]),
                             np.linalg.norm(src_points[1] - src_points[2])))

        # Destination points for the transformed image
        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        warped_img = cv2.warpPerspective(image, matrix, (max_width, max_height))

        return warped_img

    except Exception as e:
        print(f"Error in transform: {e}")
        return image




