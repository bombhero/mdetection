import cv2
import numpy as np


class MotionDetection:
    def __init__(self, seq_len=2, block_size=20, sen_rate=0.8):
        self.frame_seq = []
        self.seq_len = seq_len
        self.block_size = block_size
        self.sen_rate = sen_rate

    def motion_detect(self, image):
        detect_result = []
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (self.block_size*2+1, self.block_size*2+1), 0)
        cv2.imshow('gray', gray_img)
        self.frame_seq.append(gray_img)
        if len(self.frame_seq) > self.seq_len:
            self.frame_seq.remove(self.frame_seq[0])
        else:
            return detect_result
        diff_img = np.abs(self.frame_seq[-1] - self.frame_seq[0])
        y_idx = 0
        while y_idx < diff_img.shape[0]:
            if y_idx + self.block_size > diff_img.shape[0]:
                y_end = diff_img.shape[0]
            else:
                y_end = y_idx + self.block_size
            x_idx = 0
            while x_idx < diff_img.shape[1]:
                if x_idx + self.block_size > diff_img.shape[1]:
                    x_end = diff_img.shape[1]
                else:
                    x_end = x_idx + self.block_size
                block_diff = float(np.sum(diff_img[y_idx:y_end, x_idx:x_end])) / float(self.block_size ** 2) / 256
                if block_diff > self.sen_rate:
                    detect_result.append([(x_idx, y_idx), (x_end, y_end)])
                x_idx += self.block_size
            y_idx += self.block_size
        return detect_result


def main():
    avi_file = 'sample/vtest.avi'
    motion_detector = MotionDetection(seq_len=3, block_size=15)
    cap = cv2.VideoCapture(avi_file)
    ret, frame = cap.read()
    while ret:
        md_result = motion_detector.motion_detect(frame)
        for point in md_result:
            cv2.rectangle(frame, point[0], point[1], (255, 0, 0), 1)
        cv2.imshow('vtest', frame)
        cv2.waitKey(25)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
