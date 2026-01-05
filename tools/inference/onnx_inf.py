"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import cv2

# COCO 80-class names (index = class id)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def color_for_class(cls_id: int):
    # Deterministic bright colors
    np.random.seed(cls_id + 12345)
    color = np.random.randint(64, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def coco_name(lbl: int) -> str:
    return COCO80[lbl] if 0 <= lbl < len(COCO80) else str(lbl)

def draw_rounded_rectangle(image, top_left, bottom_right, color=(0, 255, 0),
                           thickness=5, radius_ratio=0.1, coeff=0.25):
    x1, y1 = top_left
    x2, y2 = bottom_right
    width, height = x2 - x1, y2 - y1
    radius = int(radius_ratio * max(width, height))
    thickness = min(thickness, radius if radius > 0 else thickness)

    w_d = int(width * coeff)
    h_d = int(height * coeff)

    cv2.line(image, (x1 + radius, y1), (x1 + w_d, y1), color, thickness)
    cv2.line(image, (x2 - w_d, y1), (x2 - radius, y1), color, thickness)

    cv2.line(image, (x1 + radius, y2), (x1 + w_d, y2), color, thickness)
    cv2.line(image, (x2 - w_d, y2), (x2 - radius, y2), color, thickness)

    cv2.line(image, (x1, y1 + radius), (x1, y1 + h_d), color, thickness)
    cv2.line(image, (x1, y2 - h_d), (x1, y2 - radius), color, thickness)

    cv2.line(image, (x2, y1 + radius), (x2, y1 + h_d), color, thickness)
    cv2.line(image, (x2, y2 - h_d), (x2, y2 - radius), color, thickness)

    if radius > 0:
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red')
            draw.text((bb[0], bb[1]), text=str(lbl), fill='blue')

        result_images.append(im)
    return result_images

def draw_cv2(bgr_images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    """
    bgr_images: list[np.ndarray] BGR images (OpenCV)
    labels/boxes/scores: ONNX outputs (batch-first)
    ratios: list[float]
    paddings: list[(pad_w, pad_h)]
    """
    result_images = []
    for i, img in enumerate(bgr_images):
        scr = scores[i]
        keep = scr > thrh

        lab = labels[i][keep]
        box = boxes[i][keep]
        scr = scr[keep]

        ratio = float(ratios[i])
        pad_w, pad_h = paddings[i]

        for lbl, bb, s in zip(lab, box, scr):
            lbl_i = int(lbl)  # ORT returns numpy scalar/float sometimes
            name = coco_name(lbl_i)

            # Map boxes back to original image coordinates (undo pad + scale)
            x1 = int(round((float(bb[0]) - pad_w) / ratio))
            y1 = int(round((float(bb[1]) - pad_h) / ratio))
            x2 = int(round((float(bb[2]) - pad_w) / ratio))
            y2 = int(round((float(bb[3]) - pad_h) / ratio))

            # Clip to image
            h, w = img.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # Rounded rectangle
            color = color_for_class(lbl_i)
            draw_rounded_rectangle(img, (x1, y1), (x2, y2), color=color, thickness=3)

            # Label text
            text = f"{name} {float(s):.2f}"
            # simple filled background
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(0, y1 - th - baseline - 6)
            cv2.rectangle(img, (x1, ty1), (x1 + tw + 6, ty1 + th + baseline + 6), (0, 255, 0), -1)
            cv2.putText(img, text, (x1 + 3, ty1 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        result_images.append(img)
    return result_images


def process_image(sess, im_pil):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
        T.ToTensor(),
    ])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )

    labels, boxes, scores = output

    # Convert original PIL image to OpenCV BGR
    bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    out_imgs = draw_cv2(
        [bgr], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )

    cv2.imwrite("onnx_result.jpg", out_imgs[0])
    print("Image processing complete. Result saved as 'onnx_result.jpg'.")


def process_video(sess, video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('onnx_result.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, 640)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )

        labels, boxes, scores = output
    
        # Draw directly on the original OpenCV BGR frame
        out_frames = draw_cv2(
            [frame], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        frame = out_frames[0]

        # # Convert back to OpenCV image
        # frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(f"Using device: {ort.get_device()}")

    input_path = args.input

    try:
        # Try to open the input as an image
        im_pil = Image.open(input_path).convert('RGB')
        process_image(sess, im_pil)
    except IOError:
        # Not an image, process as video
        process_video(sess, input_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    args = parser.parse_args()
    main(args)
