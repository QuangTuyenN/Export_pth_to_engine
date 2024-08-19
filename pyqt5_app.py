from rotate_license import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as fu

from PIL import Image
import datetime


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 112, 5)
        self.fc1 = nn.LazyLinear(out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 31)

    def forward(self, x):
        x = self.pool(fu.relu(self.conv1(x)))
        x = self.pool(fu.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = fu.relu(self.fc1(x))
        x = fu.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ProcessVideo:
    def __init__(self):
        self.ThreadActive = None
        self.model_track = YOLO('models/yolov8n.pt')
        self.model_detect_lp = YOLO('./models/detect_lp/lp_detect_yolov8n_seg.pt')
        self.model_detect_char = YOLO('./models/detect_char/chrs_detect_yolov8n.pt')

        self.count = 0
        self.list_license_plate = []
        self.vehicle_classes = {2: 'Car', 3: 'Mortorbike'}
        self.processed_id = []

        # define the architecture of model to load model using state dict

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.char_recog_path = './models/char_recognize/chrs_recog_cnnv6.pth'
        self.model_char_recog = Net().to(self.device)
        self.model_char_recog.load_state_dict(torch.load(self.char_recog_path))
        self.model_char_recog.eval()

        # define transform to transform data
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Grayscale(),
             transforms.Normalize(0.5, 0.5)])

        # classify char
        self.ALPHA_DICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
                           11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L', 20: 'M',
                           21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}

    def calculate_area(self, image):
        # create threshold value is 127 if pixel has value is greater than 127 it's to be 255 and if lower than 127 it's to be 0
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours of image
        areas = [cv2.contourArea(c) for c in contours]
        index = np.argmax(areas)
        return areas[index]

    def format_LP(self, chars, char_centers):
        x = [c[0] for c in char_centers]
        y = [c[1] for c in char_centers]
        y_mean = np.mean(y)
        # if all character is in one line, we only sort x center to have correct position of all characters
        # it's car license plate==>we add - in 3 position
        if y_mean - min(y, default=0) < 0.1:
            k = [i for _, i in sorted(zip(x, chars))]
            return k
        # if all character is in two lines
        sorted_chars = [i for _, i in sorted(zip(x, chars))]
        y = [i for _, i in sorted(zip(x, y))]
        # sort in line one
        first_line = [i for i in range(len(chars)) if y[i] < y_mean]
        # sort in line two
        second_line = [i for i in range(len(chars)) if y[i] > y_mean]
        # concatnate line one and line two to have number of lp
        lp_full_text = [sorted_chars[i] for i in first_line] + [sorted_chars[i] for i in second_line]
        return lp_full_text

    def run(self):
        self.ThreadActive = True
        source = "./video/test.mp4"
        results = self.tracking_vehicle(source)
        while self.ThreadActive:
            for result in results:
                self.count += 1
                frame = result.orig_img

                if self.count % 4 != 0:
                    continue
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes_vehicle = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, track_id, classid in zip(boxes_vehicle, track_ids, classes):
                    xcar1, ycar1, xcar2, ycar2 = box
                    if xcar1 <= 250 and track_id not in self.processed_id:
                        # xcar1, ycar1, xcar2, ycar2 = box
                        image = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]
                        # skip motorbike
                        if classid == 3:
                            continue
                        r = rotate_lp(image)

                        if r is None:
                            continue

                        cls = self.vehicle_classes[classid]
                        LpRegion = r['LpRegion']
                        Image2 = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2RGB)
                        Lp_image = cv2.resize(Image2, (500, 500))

                        r = self.detect_char(LpRegion)
                        if r is None:
                            continue
                        Image3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        Vehicle_image = cv2.resize(Image3, (500, 500))

                        characters = r['characters']
                        char_centers = r['char_centers']
                        license_plate_text = self.char_recognize(characters, char_centers)
                        self.processed_id.append(track_id)

                        lp_infor = {"Vehicle": Vehicle_image, "License_Plate": Lp_image,
                                    "License_Plate_No": license_plate_text,
                                    "Time": str(datetime.datetime.now()), "Vehicle_Type": cls,
                                    "Camera_Name": "Hitec_Speed_CAM1"}

                        # self.list_license_plate.append(lp_infor)
                        self.list_license_plate = [lp_infor] + self.list_license_plate

                    else:
                        continue

    def tracking_vehicle(self, source):
        return self.model_track.track(source=source, stream=True, classes=[2, 3], persist=True, conf=0.5)

    def resize_lp(self, LpRegion):
        width = 1500
        ratio = width / LpRegion.shape[1]
        w = round(LpRegion.shape[1] * ratio)
        h = round(LpRegion.shape[0] * ratio)
        new_im = cv2.resize(LpRegion, (w, h))
        height = new_im.shape[0] + 100
        border = height - new_im.shape[0]
        if border % 2 == 0:
            border_T = border_B = border // 2
        else:
            border_T = border // 2
            border_B = border_T + 1
        new_im = cv2.copyMakeBorder(new_im, border_T, border_B, 0, 0, cv2.BORDER_CONSTANT, (255, 255, 255))
        LpRegion = new_im
        return LpRegion

    # detect char in lp
    def detect_char(self, LpRegion):
        list_areas = []
        bboxes_yolo = []
        characters = []
        results = self.model_detect_char.predict(LpRegion, iou=0.5)
        boxes = results[0].boxes
        if len(results[0].boxes) == 0:
            return None
        for box in boxes:
            # loop all boxes of all characters is detected by model
            boxa = box.xyxy.cpu().numpy()[0]
            xmin = int(boxa[0])
            ymin = int(boxa[1])
            xmax = int(boxa[2])
            ymax = int(boxa[3])
            character = LpRegion[ymin:ymax, xmin:xmax]
            # convert image to gray scale
            im = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
            # using gausian adaptive threshold to process image on each small region
            im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5)
            # remove the noise in the border of license plate
            if (1500 - xmax <= 10 and box.conf < 0.8) or (xmin <= 10 and box.conf < 0.8):
                continue
            # update code to change color for blue, red and yellow license plate
            average_color = np.mean(im)
            threshold = 199
            if average_color > threshold:
                im = cv2.bitwise_not(im)
            list_areas.append(self.calculate_area(im))
            # resize to 224*224
            ratio = 224 / im.shape[0]
            w = round(im.shape[1] * ratio)
            h = round(im.shape[0] * ratio)
            im = cv2.resize(im, (w, h))
            if im.shape[1] > 224:
                im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
                new_im = im
            else:
                border = 224 - im.shape[1]
                if border % 2 == 0:
                    border_L = border_R = border // 2
                else:
                    border_L = border // 2
                    border_R = border_L + 1
                new_im = cv2.copyMakeBorder(im, 0, 0, border_L, border_R, cv2.BORDER_CONSTANT,
                                            (255, 255, 255))
            box1 = box.xywhn.cpu().numpy()[0]
            bboxes_yolo.append(box1)
            # save all characters to predict
            characters.append(new_im)
            # using char center to order character position in lp
        char_centers = [bboxes_yolo[i][:2] for i in range(len(bboxes_yolo))]
        # remove black hole in license plate
        while len(characters) > 9:
            index = np.argmin(list_areas)
            characters.pop(index)
            char_centers.pop(index)
            list_areas.pop(index)
        r = {'characters': characters, 'char_centers': char_centers}
        return r

    # char recoginize
    def char_recognize(self, characters, char_centers):
        chars = []
        # loop all characters to recognize
        for i in range(len(characters)):
            # predict without gradient descent
            with torch.no_grad():
                # convert array to pil image
                pil_image = Image.fromarray(characters[i])
                # transform image and move it to GPU top process
                image = self.transform(pil_image).to(self.device)
                image = image.unsqueeze(0)
                # uisng forward function to predict
                output = self.model_char_recog.forward(image)
                # get probabilities of each class
                probabilities = fu.softmax(output, dim=1)
                # get the index of max probabilities
                index = np.argmax(probabilities.cpu().numpy(), axis=1)
                # mapping index with ALPHA_DICT to get predicted char and append it to get all value in lp
                chars.append((self.ALPHA_DICT[index[0]]))
        # get the correct format of number plates
        license_plate_text = ''.join(self.format_LP(chars, char_centers))
        return license_plate_text


if __name__ == "__main__":
    pass
