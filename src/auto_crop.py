import cv2

class ParamCropper:

    def __init__(self, center_x=None, center_y=None, ratio_w=0.6, ratio_h=0.6, output_size=(800,800)):

        self.cx = center_x
        self.cy = center_y

        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.output_size = output_size

        # centre verrouillé (pour stabiliser le crop dans la vidéo)
        self.locked_center = None


    def crop_frame(self, frame):

        h, w = frame.shape[:2]

        # si le centre n'est pas encore verrouillé
        if self.locked_center is None:

            if self.cx is None:
                self.cx = w // 2
            if self.cy is None:
                self.cy = h // 2

            self.locked_center = (self.cx, self.cy)

        cx, cy = self.locked_center

        crop_w = int(w * self.ratio_w)
        crop_h = int(h * self.ratio_h)

        x1 = max(0, cx - crop_w//2)
        x2 = min(w, cx + crop_w//2)

        y1 = max(0, cy - crop_h//2)
        y2 = min(h, cy + crop_h//2)

        cropped = frame[y1:y2, x1:x2]

        resized = cv2.resize(cropped, self.output_size)

        return resized, x1, y1, crop_w