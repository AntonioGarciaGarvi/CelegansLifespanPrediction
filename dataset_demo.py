import torch
import cv2
import os
from scipy.optimize import curve_fit
import numpy as np
import re
from PIL import Image
import random

##Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# def get_movement1(object_no,object_tot):
#     survival = object_no/object_tot
#     if survival >= 0.5:
#         return random.randint(20, 30)
#     else:
#         return random.randint(2, 12)

def post_process(real_labels, filter_limit=False, filter_correction=False):

    processed_labels = []
    if filter_limit: #The maximum number (or limit) of the number of living of worms is the number of worms that were living in the first day
        living_worms_first_day = max(real_labels)
        for lab in range(len(real_labels)):
            processed_labels.append(min(real_labels[lab], living_worms_first_day))
    if filter_correction:
        for lab2 in range(len(real_labels)):
            x = processed_labels[lab2]
            i = 1
            while lab2-i >= 0 and x > processed_labels[lab2-i]:
                processed_labels[lab2-i] = x
                i += 1
    p = [int(round(a)) for a in processed_labels]
    return p


def get_movement(object_no, object_tot):
    survival = object_no / object_tot
    if survival >= 0.66:
        return random.randint(1, 200) #random.randint(100, 200)

    elif survival >= 0.33:
        return random.randint(1, 30) #random.randint(20, 30)

    else:
        return random.randint(1, 12)


#
# def get_overlap_prob(object_no,object_tot):
#     survival = object_no/object_tot
#     if survival >= 0.66:
#         return 0.5
#
#     elif survival >= 0.33:
#         return 0.1
#     else:
#         return 0.01

def get_overlap_prob(object_no,object_tot):
    survival = object_no/object_tot
    return 0.5
    # if survival >= 0.66:
    #     return 0.5
    #
    # elif survival >= 0.33:
    #     return 0.1
    # else:
    #     return 0.01


def random_point_inside_circle(cx, cy, rad):
    r = rad * np.sqrt(random.random())
    theta = random.random() * 2 * np.pi
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return int(x), int(y)

def random_point_in_circumference(cx, cy, rad):
    theta = random.randint(0, 360)
    theta = theta * np.pi /180
    x = cx + rad * np.cos(theta)
    y = cy + rad * np.sin(theta)
    return int(x), int(y)

def generate_curve_weibull(day_init, n_objects, stepness, mean_life):
    # print(stepness,mean_life)
    alive = n_objects
    distribution = []
    curve = []
    days = []
    t = day_init

    while alive > 0:
        survival = np.exp(-(t / mean_life) ** stepness)
        distribution.append(survival)
        alive = int(round((survival*n_objects)))
        curve.append(alive)
        days.append(t)
        t = t + 1

    dead_curve = [n_objects - x for x in curve]
    return curve, dead_curve, days, distribution


def calculateDistance(x1, y1, x2, y2):
    dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def relativeposCircles(x1, y1, r1, x2, y2, r2):
    dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # no overlap
    if dist > r1 + r2 :
        return 0, None

    # one circle inside another
    elif dist < abs(r1 - r2):
        if r1 < r2:
            xc = x1
            yc = y1
            rc = r1
        else:
            xc = x2
            yc = y2
            rc = r2


        return 1, (xc, yc, rc)
    # coincident circles
    elif dist == 0 and r2 == r1:
        return 0, None
    # intersection
    else:
        a = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)
        h = sqrt(r1 ** 2 - a ** 2)
        x3 = x1 + a * (x2 - x1) / dist
        y3 = y1 + a * (y2 - y1) / dist

        x4 = x3 + h * (y2 - y1) / dist
        y4 = y3 - h * (x2 - x1) / dist

        x5 = x3 - h * (y2 - y1) / dist
        y5 = y3 + h * (x2 - x1) / dist

        return 2, (x4, y4, x5, y5)

# define the true objective function
def objective(x,  a,  b):
    # np.exp(-(t / mean_life) ** stepness)
    return np.exp(-(x / b) ** a)




class LifespanDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root_dir, seq_length, transform=None, augmentation=None, index_stop=0):
        self.root_dir = root_dir
        self.seq_lenght = seq_length
        self.transform = transform
        self.augmentation = augmentation
        self.index_stop = index_stop

    def __len__(self):
        return int(len(os.listdir(self.root_dir)))

    def __getitem__(self,idx):

        subdir = self.root_dir.split('/')[-1]
        labels = torch.zeros(self.seq_lenght)
        list_labels = []
        list_pics = []
        folders_days = os.listdir(self.root_dir)
        folders_days.sort(key=natural_keys)

        days_available = []
        for sub_days_dir in folders_days:
            start = sub_days_dir.find("dia=")
            finish = sub_days_dir.find(" cond")
            days_available.append(int(sub_days_dir[start + len("dia="):finish]))

        # Se eliminan las imágenes del día 1
        if days_available[0] == 1:
            del(folders_days[0])


        day_real = 4
        day = 0

        for file in folders_days:
            start = file.find("dia=")
            finish = file.find(" cond")
            day_available = int(file[start + len("dia="):finish])

            if day_real != day_available:
                day_before_gap = day_real - 1
                day_after_gap = days_available[days_available.index(day_before_gap) + 1]
                gaps = day_after_gap - day_before_gap - 1

                for gap in range(gaps):
                    img = 255 * np.ones((256, 256), np.uint8)
                    img = Image.fromarray(img)
                    img_tens = self.transform(img)
                    list_pics.append(img_tens)
                    list_labels.append(int(label_viv))
                    labels[day] = no_living_worms
                    day += 1
                    day_real += 1

            img = cv2.imread(self.root_dir + "/" + file, 0)

            start = file.find("living_worms=")
            finish = file.find(".jpg")
            label_viv = (file[start + len("living_worms="):finish])

            img = Image.fromarray(img)
            img_tens = self.transform(img)
            list_pics.append(img_tens)
            no_living_worms = torch.tensor(np.asarray([int(label_viv)]), dtype=torch.int)
            labels[day] = no_living_worms
            list_labels.append(int(label_viv))

            day += 1
            day_real += 1

        # En las secuencias que son más cortas se replica la última imagen y se pone la etiqueta a 0
        if len(list_pics) < self.seq_lenght:
            for rep in range(0, (self.seq_lenght - len(list_pics))):
                list_pics.append(img_tens)
                no_living_worms = torch.tensor(np.asarray([int(0)]), dtype=torch.int)
                labels[day + rep] = no_living_worms
                list_labels.append(0)

        list_labels = post_process(list_labels, filter_limit=True, filter_correction=True)
        list_labels.extend([0] * (self.seq_lenght - len(list_labels)))

        stacked_set = torch.stack(list_pics)

        try:
            duration = next(v for v in range(len(list_labels)) if list_labels[v] == 0) # el ensayo termina cuando no quedan gusanos vivos

        except Exception:
            duration = len(list_labels)

        if duration > self.seq_lenght:
            duration = self.seq_lenght

        if self.index_stop == self.seq_lenght - 1:
            self.index_stop = self.seq_lenght - 2

        if self.index_stop >= self.seq_lenght:
            self.index_stop = self.seq_lenght

        if self.index_stop == 0:
            self.index_stop = 1

        labels = torch.FloatTensor(list_labels)
        stacked_set = stacked_set[0:self.index_stop] # imágenes de entrada

        input_counts = labels[0:self.index_stop]

        future_counts = labels[self.index_stop:duration]

        composed_sample = [stacked_set, input_counts, future_counts, subdir]

        return composed_sample