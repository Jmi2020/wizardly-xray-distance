import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

def find_red_dots(image_path, dot_diameter):
    """Find red dots in the image and return their centroids."""
    img = cv2.imread(image_path)
    if img is None:
        return [], "Image not loaded. Check the file path."
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    expected_area = np.pi * (dot_diameter / 2) ** 2
    area_range = (expected_area * 0.5, expected_area * 1.5)
    
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_range[0] < area < area_range[1]:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
    
    centroids.sort(key=lambda x: x[1])
    return centroids, None

def calculate_vertical_distances(centroids):
    """Calculate vertical distances between consecutive centroids."""
    vertical_distances = []
    for i in range(len(centroids) - 1):
        dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i + 1]))
        vertical_distances.append(dist)
    return vertical_distances

def browse_image():
    """Open a file dialog to select an image and process it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        centroids, error = find_red_dots(file_path, dot_diameter=25)
        if error:
            result_text.insert(tk.END, error + '\n', 'error')
            return
        
        vertical_distances = calculate_vertical_distances(centroids)
        result_text.delete('1.0', tk.END)
        for i, dist in enumerate(vertical_distances):
            tag = f"color_{i % 2}"
            result_text.insert(tk.END, f"Distance between dot {i+1} at {centroids[i]} and dot {i+2} at {centroids[i+1]}: {dist:.2f} pixels.\n", tag)
        
        root.geometry('')

def setup_gui():
    """Set up the Tkinter GUI."""
    global root, result_text
    root = tk.Tk()
    root.title("Dot Distance Calculator")
    
    browse_button = tk.Button(root, text="Browse Image", command=browse_image)
    browse_button.pack(pady=10)
    
    result_text = tk.Text(root, height=20, width=80)
    result_text.pack(expand=True, fill='both')
    
    scrollbar = tk.Scrollbar(root, command=result_text.yview)
    scrollbar.pack(side='right', fill='y')
    result_text['yscrollcommand'] = scrollbar.set
    
    result_text.tag_configure('color_0', background='#FFF68F')
    result_text.tag_configure('color_1', background='#E0FFFF')
    result_text.tag_configure('error', background='#FF6A6A')
    
    root.mainloop()

if __name__ == "__main__":
    setup_gui()
