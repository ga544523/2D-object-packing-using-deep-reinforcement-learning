import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import cmath
import math
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import random
import threading
import heapq as hq
import time
from multiprocessing import Process, Queue
from multiprocessing import Process, Array
import os
from multiprocessing import Pool
import multiprocessing
from numba import jit, njit
import matplotlib.pyplot as plt
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")
device3 = torch.device("cuda:2")
device4 = torch.device("cuda:3")


def savetensor(tensor, filename, ch=0, allkernels=False, nrow=8, padding=2):
   '''
   savetensor: save tensor
       @filename: file name
       @ch: visualization channel
       @allkernels: visualization all tensores
   '''


   n, c, w, h = tensor.shape


   print(tensor.shape)
   if allkernels:
       tensor = tensor.view(n * c, -1, w, h)
   elif c != 3:
       tensor = tensor[:, ch, :, :].unsqueeze(dim=1)


   print(tensor.shape)
   utils.save_image(tensor, filename, nrow=nrow, normalize=True)

def outt():
   ik =0
   kernel = net222.features[ik].weight.data.clone()
   print(len(kernel) )

   for i in range(0,64,1):
       for j in range(0,11,1):
           for k in range(0,11,1):
               if(kernel[i][0][j][k]>0):
                   kernel[i][0][j][k]+=100
                   print(kernel[i][0][j][k])
   savetensor(kernel, 'kernelshoe.png', allkernels=False)
   print(1/0)

def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x


def cross(p1, p2, p0):
    return ((p1[0] - p0[0]) * (p2[1] - p0[1])) - ((p2[0] - p0[0]) * (p1[1] - p0[1]))


__all__ = ['alexnet']

devicecpu = torch.device("cpu")
if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)


def default_loader(path):
    return Image.open(path).convert('L')


import torch
import torch.nn as nn

# 80 256
# 200 6400
# 400 30976
# 300 16384
W = 200
seperate = 6400
WW = W * W
searchsplit = 50


class AlexNet(nn.Module):

    def __init__(self, num_classes=200 * 200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(device1)

        self.classifier = nn.Sequential(
            # 4096
            nn.Linear(seperate, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3 * searchsplit),
        ).to(device1)

    def forward(self, x):
        x = self.features(x)
        x = x.to(device1)
        x = x.view(x.size(0), seperate)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    return model


device = torch.device("cuda:1")

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


if __name__ == '__main__':
    net222 = alexnet().to(device1)

    # transforms.ToTensor()
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )

    overlapnum = []


def checkoverlap(imgoverlap, x, y, rotation, now, first, nowshape, overlapnum):
    if (first == 1):
        return 0


    for i in range(0, len(now), 1):
        xx = int(now[i][0])
        yy = int(now[i][1])
        if (xx >= W or xx < 0 or yy >= W or yy < 0 or imgoverlap[yy, xx] == 255):
            return 1

    k = int(rotation)
    for i in range(0, len(overlapnum[nowshape][k]), 1):
        if (overlapnum[nowshape][k][i][0] == -1):
            continue
        newi = int(y - int(W / 2) + overlapnum[nowshape][k][i][0])
        newj = int(x - int(W / 2) + overlapnum[nowshape][k][i][1])
        if imgoverlap[newi][newj] == 255:

            # 列印結果

            return 1

    # 列印結果

    return 0


def drawimg(imgdraw, x, y, rotation, now):
    now = np.array(now)
    tmpimgdraw = imgdraw.copy()
    mask = np.zeros((W + 2, W + 2), np.uint8)
    cv2.polylines(tmpimgdraw, [now], True, (255, 255, 255), thickness=1)

    cv2.floodFill(tmpimgdraw, mask, (int(x), int(y)), (255, 255, 255), (0,), (0,), 4)  # 填充封??域
    return tmpimgdraw


def collect(arr, nowshape):
    global featureshape
    global overallimg
    global overalllabels
    global tttttttttt
    global yyyyyyyyyy
    i = 0

    k = len(arr)

    while (i < k):
        # get the inputs

        inputs = torch.from_numpy(arr[i][0])  # ?一化到 [0.0,1.0]
        inputs = inputs.view(1, 1, W, W)
        label = []

        # 按下任意鍵則關閉所有視窗

        firx = -1
        firy = -1
        firr = -1

        for j in range(0, len(arr[i][1]), 1):
            buf = int(arr[i][1][j][1] + 0.5)

            r = int(buf / WW)
            buf %= WW
            y = int(buf / W)
            buf %= W
            x = buf
            label.append(x)
            label.append(y)
            label.append(r)
            if (firx < 0):
                firx = x
                firy = y
                firr = r

        while (len(label) < 3 * searchsplit):
            label.append(firx)
            label.append(firy)
            label.append(firr)

        if (tttttttttt == 0):
            overallimg = inputs
            tttttttttt = 1
        else:
            overallimg = torch.cat((overallimg, inputs), 0)

        labels = convert2tensor(label)
        labels = labels.view(1, -1)
        if (yyyyyyyyyy == 0):
            overalllabels = labels
            yyyyyyyyyy = 1
        else:
            overalllabels = torch.cat((overalllabels, labels), 0)
        i = i + 1
        overallimg = torch.cat((overallimg, featureshape), 0)


def retrain():
    criterion = nn.MSELoss()
    global net222
    optimizer = optim.Adam(net222.parameters(), lr=0.001, betas=(0.9, 0.999))

    global overallimg
    global overalllabels
    overallimg = overallimg.view(-1, 2, W, W)
    overallimg = overallimg.float()
    overallimg, overalllabels = overallimg.to(device1), overalllabels.to(device1)
    print(overallimg.size())
    print(overalllabels)
    maxloss = 10000000
    tStart = time.time()  # 計時開始
    for epoch in range(200):  # loop over the dataset multiple times
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net222(overallimg)

        loss = criterion(outputs, overalllabels)

        loss.backward()
        optimizer.step()
        '''
        if (maxloss / 1.2 > loss.item()):
            maxloss = loss.item() 
            print(maxloss)
        '''

    torch.save(net222, 'net 200 select update with move shoe2 50 out.pkl')
    tEnd = time.time()  # 計時結束
    # 列印結果
    print('It cost sec training" ', tEnd - tStart)  # 會自動做近位

@jit(nopython=True)
def GetAreaOfPolyGon(points):
    '''
    class Point():
        def __init__(self, x, y):
            self.x = x
            self.y = y
    '''

    area = 0
    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[i]
        p3 = points[i + 1]
        vecp1p2 = [p2[0] - p1[0], p2[1] - p1[1]]  # Point(p2[0] - p1[0], p2[1] - p1[1])
        vecp2p3 = [p3[0] - p2[0], p3[1] - p2[1]]  # (p3[0] - p2[0], p3[1] - p2[1])
        vecMult = vecp1p2[0] * vecp2p3[1] - vecp1p2[1] * vecp2p3[0]
        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1
        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


@jit(nopython=True)
def GetAreaOfTriangle(p1, p2, p3):
    area = 0
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area


@jit(nopython=True)
def GetLineLength(p1, p2):
    length = math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)
    length = math.sqrt(length)
    return length


def savenotglobal(res):
    global maxlist
    global countmax
    maxlist = list(res)
    countmax = len(res)


def getglobal():
    global countmax
    return countmax


def getglobaltest(imgsearch):
    return test(imgsearch)


@jit(nopython=True)
def rotate(x, y, rotation, onlyonenow):
    newobjectlist = []
    index = 0
    n = len(onlyonenow)
    if (rotation >= 180):
        rotation -= 360;

    if (rotation < -180):
        rotation += 360;

    while (index < n):
        R = math.sqrt(onlyonenow[index][0] * onlyonenow[index][0] + onlyonenow[index][1] * onlyonenow[index][1])
        rot = math.atan2(onlyonenow[index][1], onlyonenow[index][0]) - (rotation / 180 * 3.1415926)

        tx = int(R * math.cos(rot))
        ty = int(R * math.sin(rot))
        newobjectlist.append([tx + x, ty + y])
        index = index + 1
    return newobjectlist

def threadgenerate(x1, x2, y1, y2, imgsearch, grouplist, pointsize, nx, ny, nr, timeofsearch, d, id, currentpoint,
                   onlyonenow, nowshape, overlapnum, x, y, rotation,rrrr):
    global heapthread
    heap = []
    prex=-1000
    prey=-1000
    prer=-1000

    for abc in range(0,6,1):
        for i in range(0, searchsplit, 1):
            # 按下任意鍵則關閉所有視窗
            nx = x[i]
            ny = y[i]
            nr = rotation[i] + 5*abc+rrrr
            nr %= 360

            same=abs( prex-nx )+abs( prey-ny )+abs( prer-nr )
            if(same<=30):
                continue

            prex=nx
            prey=ny
            prer=nr

            globarea = 99999999999
            for left in range(x1, x2, 1):
                carea = 99999999
                for right in range(y1, y2, 1):
                    nnx = nx + left
                    nny = ny + right
                    if(nnx>=W):
                        nnx%=W
                    if(nny>=W):
                        nny%=W
                    if (nnx > 0 and nny > 0):
                        nowres = rotate(nnx, nny, nr, onlyonenow)

                        no = 1
                        no = checkoverlap(imgsearch, nnx, nny, nr, nowres, 0, nowshape, overlapnum)


                        if (no == 1):
                            continue
                        if (no == 0):
                            nk = len(nowres)
                            summ=0
                            for j in range(nk):
                                currentpoint.append(nowres[j][0])
                                currentpoint.append(nowres[j][1])
                                summ+=nowres[j][0]
                                summ+=nowres[j][1]
                            myarray = np.asarray(currentpoint)
                            myarray = myarray.reshape(-1, 2)
                            point = cv2.convexHull(myarray)
                            myarray = np.asarray(point)
                            myarray = myarray.reshape(-1, 2)
                            area = GetAreaOfPolyGon(myarray)
                            if (d == 0):
                                area = nnx + nny
                            hq.heappush(heap, ( (-area- summ) , nr * WW + nny * W + nnx) )
                            if (len(heap) > searchsplit):
                                hq.heappop(heap)
                            for rr in range(pointsize):
                                currentpoint.pop()
                                currentpoint.pop()
                            if (carea < area):
                                break
                            carea = area
                if (globarea < carea):
                    break
                globarea = carea
    small = hq.nlargest(searchsplit, heap)
    return small

nodeindex=0
def search(imgsearch, d, deep, nowshape, timeofsearch, minarea, res, lim, currentpoint, grouplist, pointsize,
           currentarea):
    global i_want_to_save_currentpoint
    global outputimage
    global nodeindex
    '''
    if(d==9):
        return
    '''

    if (d == deep or timeofsearch[0][0] >= lim):
        if (len(res) > getglobal()):
            minarea[0][0] = currentarea
            savenotglobal(res)
            i_want_to_save_currentpoint = list(currentpoint)
            outputimage = imgsearch.copy()
        if (len(res) == getglobal() and currentarea < minarea[0][0]):
            minarea[0][0] = currentarea
            savenotglobal(res)
            i_want_to_save_currentpoint = list(currentpoint)
            outputimage = imgsearch.copy()
        return
    heap = []
    array = test(imgsearch)
    if (timeofsearch[0][0] >= lim):
        return
    x = [0] * 105
    y = [0] * 105
    rotation = [0] * 105

    for ii in range(int(0), int(1000), 1):
        timeofsearch[0][0] = timeofsearch[0][0] + 1
        oktime = 0
        if (timeofsearch[0][0] >= lim):
            return
        nx = random.randint(int(0), int(W))
        ny = random.randint(int(0), int(W))
        summ = 0
        if (len(grouplist) == 1):
            nr = random.randint(0, 359)
        if (nr < 0):
            nr = nr + 360
        for jj in range(0, len(grouplist), 1):
            nnx = nx + grouplist[jj][0]
            nnx %= W
            nny = ny + grouplist[jj][1]
            nny %= W
            if (len(grouplist) != 1):
                nr = grouplist[jj][2] % 360
            nr %= 360
            if (nnx > 0 and nny > 0):
                nowres = rotate(nnx, nny, nr, onlyonenow)
                # 按下任意鍵則關閉所有視窗
                no = 1
                if (len(nowres) > 1):

                    no = checkoverlap(imgsearch, nnx, nny, nr, nowres, 0, nowshape, overlapnum)

                if (no == 1):
                    break

                if (no == 0 and len(nowres) != 0):
                    nk = len(nowres)
                    for j in range(nk):
                        currentpoint.append(nowres[j][0])
                        currentpoint.append(nowres[j][1])
                        summ += nowres[j][0]
                        summ += nowres[j][1]
                    oktime += 1
        if (oktime != len(grouplist)):
            for rrr in range(0, oktime, 1):
                for rr in range(0, pointsize, 1):
                    currentpoint.pop()
                    currentpoint.pop()
        if (oktime == len(grouplist)):
            myarray = np.asarray(currentpoint)
            myarray = myarray.reshape(-1, 2)
            point = cv2.convexHull(myarray)
            myarray = np.asarray(point)
            myarray = myarray.reshape(-1, 2)
            area = GetAreaOfPolyGon(myarray)
            if (d == 0):
                area = nnx + nny
            hq.heappush(heap, (area+summ, nr * WW + nny * W + nnx))
            for rrr in range(0, len(grouplist), 1):
                for rr in range(pointsize):
                    currentpoint.pop()
                    currentpoint.pop()

    # 列印結果

    for i in range(0, searchsplit, 1):
        rotation[i] = int(array[i * 3 + 2])
        y[i] = int(array[i * 3 + 1])
        x[i] = int(array[i * 3])
    process = [0] *15
    threadnumber = 1
    gap = int(96 / threadnumber)
    tStart = time.time()  # 計時開始
    index = 0
    processnumber=6
    pool = Pool(processnumber)
    convexpoint=[]

    if(len(currentpoint)>0):
        myarray = np.asarray(currentpoint)
        myarray = myarray.reshape(-1, 2)
        convexpoint = cv2.convexHull(myarray)
        convexpoint = convexpoint.reshape(-1)


    for index in range(0,processnumber,1):
        process[index]=pool.apply_async(threadgenerate, args=(
        -48 + (0) * gap, (-48 + (0 + 1) * gap), -48, 48, imgsearch, grouplist, pointsize, nx, ny, nr,
        timeofsearch, d, 0, list(convexpoint), onlyonenow, nowshape, overlapnum, x, y,
        rotation,index*30))

    pool.close()
    pool.join()
    for i in range(0,processnumber,1):
        tmpbuf=process[i].get()
        for j1 in range(0, len(tmpbuf), 1):
            hq.heappush(heap, (-tmpbuf[j1][0], tmpbuf[j1][1]))
    tEnd = time.time()  # 計時結束
    # 列印結果
    print('It cost sec" ', tEnd - tStart)  # 會自動做近位

    timeofsearch[0][0] += 100000
    print(timeofsearch[0][0])
    small = hq.nsmallest(searchsplit, heap)
    for i in range(0, min(len(small), 5), 1):

        ran=random.randint(int(0), int( len(small)-1 )   )
        timeofsearch[0][0] += 1
        if (timeofsearch[0][0] >= lim):
            return
        ranint=random.randint(int(0), int(10))
        buf = int(small[i][1])
        tr = int(buf / WW)
        buf %= WW
        ty = int(buf / W)
        buf %= W
        tx = int(buf)
        if(ranint<=-1):
            print(ran)
            buf = int(small[ran][1])
            tr = int(buf / WW)
            buf %= WW
            ty = int(buf / W)
            buf %= W
            tx = int(buf)
        nowt = rotate(tx, ty, tr, onlyonenow)
        nextimg = drawimg(imgsearch, tx, ty, tr, nowt)
        #cv2.imshow(str(deep)+' '+str(i), nextimg)

        '''
        if(d>=6):

            beststring = 'D:\\multitreenode\\ '
            beststring+=str(d+1)+' '+str(nodeindex)
            beststring += '.png'
            cv2.imwrite(beststring, nextimg)
            if(nodeindex==0):
                cv2.imwrite('D:\\multitreenode\\or.png', imgsearch)
            nodeindex=nodeindex+1
        '''

        nk = len(nowt)
        for j in range(nk):
            currentpoint.append(nowt[j][0])
            currentpoint.append(nowt[j][1])
        res.append([imgsearch.copy(), small])
        search(nextimg, d + 1, deep, nowshape, timeofsearch, minarea, res, lim, currentpoint, grouplist, pointsize,small[i][0])
        for rrr in range(0, len(grouplist), 1):
            for rr in range(pointsize):
                currentpoint.pop()
                currentpoint.pop()
            res.pop()


    if (len(res) > getglobal()):
        minarea[0][0] = currentarea
        savenotglobal(res)
        i_want_to_save_currentpoint = list(currentpoint)
        outputimage = imgsearch.copy()
    if (len(res) == getglobal() and currentarea < minarea[0][0]):
        minarea[0][0] = currentarea
        savenotglobal(res)
        i_want_to_save_currentpoint = list(currentpoint)
        outputimage = imgsearch.copy()

def test(imgtest):
    imgtest = torch.from_numpy(imgtest)

    imgtest = imgtest.view(1, 1, W, W)

    imgtest = torch.cat((imgtest, featureshape), 1)

    imgtest = imgtest.float()

    prediction = net222(imgtest.to(device1))

    tmp = prediction.data

    tmp = tmp.view(-1)

    id = tmp.tolist()

    return id


def changeshape(nowshape, nextgroupshape, featurecombo):
    global onlyonenow
    global featureshape
    global takethepoint
    if (nowshape == 2):
        onlyonenow = np.array(
            [-7, 0, -6, 16, -3, 28, 3, 36, 11, 36, 16, 30, 18, 20, 18, 10, 16, -10, 16, -17, 16, -33, 13, -39, 7, -40,
             0, -36, -2, -31, -4, -12
             ],
            np.int32)

    if (nowshape == 1):
        onlyonenow = np.array(
            [-7, 0, -6, 16, -3, 28, 3, 36, 11, 36, 16, 30, 18, 20, 18, 10, 16, -10, 16, -17, 16, -33, 13, -39, 7, -40,
             0, -36, -2, -31, -4, -12
             ],
            np.int32)
    if (nowshape == 0):
        #-36.05 ,-9.70, -11.09, 9.70, 36.05, -9.70 小三角
        #-15, -15, 0, 15, 20, -20, 0, -25 四邊
        onlyonenow = np.array(
            [-7, 0, -6, 16, -3, 28, 3, 36, 11, 36, 16, 30, 18, 20, 18, 10, 16, -10, 16, -17, 16, -33, 13, -39, 7, -40,
             0, -36, -2, -31, -4, -12
             ],
            np.int32)

    onlyonenow = onlyonenow.reshape(-1, 2)
    takethepoint = len(onlyonenow)
    nextimg = imgs

    for i in range(0, featurecombo, 1):

        now = rotate(int(W / 2) + nextgroupshape[i][0], int(W / 2) + nextgroupshape[i][1], nextgroupshape[i][2],
                     onlyonenow)
        flag = 0

        for j in range(0, len(now), 1):
            if (now[j][0] >= W or now[j][0] < 0 or now[j][1] >= W or now[j][1] < 0):
                flag = 1

        if (flag == 0):
            nextimg = drawimg(nextimg, int(W / 2) + nextgroupshape[i][0], int(W / 2) + nextgroupshape[i][1],
                              nextgroupshape[i][2], now)

    featureshape = torch.from_numpy(nextimg)
    featureshape = featureshape.view(1, -1, W, W)


def getrotate(overlapnum, nowshape):
    overlapnum.append([])
    for i in range(0, 360, 1):
        creat = np.zeros((W, W, 1), np.uint8)
        nowres = rotate(W / 2, W / 2, i, onlyonenow)
        nowres = np.array(nowres)
        nowres = nowres.astype(int)
        cv2.polylines(creat, [nowres], True, (255, 255, 255), thickness=1)
        mask = np.zeros((W + 2, W + 2), np.uint8)
        #cv2.floodFill(creat, mask, (int(W / 2), int(W / 2)), (255, 255, 255), (0,), (0,), 4)  # 填充封??域
        overlapnum[nowshape].append([])
        for j in range(0, W, 1):
            for k in range(0, W, 1):
                if creat[j][k] == 255:
                    overlapnum[nowshape][i].append([j, k])


def searchgroupglobal(res, i):
    global nextgroup
    nextgroup[i] = list(res)


def searchgroup(depth, combo, nowshape, maxlist1, currentpoint, vis, nex, minarea, res):
    if (depth == combo):

        myarray = np.asarray(currentpoint)
        myarray = myarray.reshape(-1, 2)

        point = cv2.convexHull(myarray)

        myarray = np.asarray(point)
        myarray = myarray.reshape(-1, 2)

        area = GetAreaOfPolyGon(myarray)
        if (area < minarea[0][0]):
            searchgroupglobal(res, combo)
            minarea[0][0] = area
        return

    for i in range(nex, len(maxlist1), 1):
        if (vis[i] == 0):
            vis[i] = 1
            res.append([maxlist1[i][0], maxlist1[i][1], maxlist1[i][2]])
            nowres = allsinglepoly[nowshape][maxlist1[i][0]][maxlist1[i][1]][maxlist1[i][2]]
            for j in range(0, len(nowres), 1):
                currentpoint.append(nowres[j][0])
                currentpoint.append(nowres[j][1])

            searchgroup(depth + 1, combo, nowshape, maxlist1, currentpoint, vis, i + 1, minarea, res)
            vis[i] = 0;
            for j in range(len(nowres)):
                currentpoint.pop()
                currentpoint.pop()
            res.pop()


if __name__ == '__main__':
    onlyonenow = np.array(
        [-15, -15, 0, 15, 20, -20, 0, -25
         ],
        np.float32)
    featureshape = 0
    imgs = np.zeros((W, W, 1), np.uint8)

    overallimg = 0
    overalllabels = 0
    tttttttttt = 0
    yyyyyyyyyy = 0

    number = 1

    maxcnt = 0
    needtime = 10000000000

    countmax = 0

    okk = 0

    countcnt = 0
    maxlist = []

    best = 0
    combo = 1
    ggg = 0

    nextgroup = [[], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]

    i_want_to_save_currentpoint = []

    outputimage = np.zeros((W, W, 1), np.uint8)
    heapthread = [0] * 110

    takethepoint = 0
    maxlistforgroup = 0
    getinit = 0
    premaxlist = []
    prepremaxlist = []
    preprepremaxlist = []
    prepreprepremaxlist=[]
    preprepreprepremaxlist=[]
    prepreprepreprepremaxlist=[]
    #net222 = torch.load('net 200 select update with move shoe1 50 out.pkl')

    ttt = 1
    while (ggg < 1000000):
        lim = 6000000
        x = []
        flag = 0
        cnt = 0
        tttttttttt = 0
        yyyyyyyyyy = 0
        maxlist = []
        nowshape = 0



        while (cnt < 1):

            if (getinit == 0):
                getinit = 1
                for i in range(0, 1, 1):
                    changeshape(i, nextgroup[1], 1)
                    print(takethepoint)
                    getrotate(overlapnum, i)

            searchshow = imgs
            objectsum = 0

            minarea = 1000000
            maxlist = []
            currentpoint = []
            result = []
            pre = 0
            result = []
            i_want_to_save_currentpoint = []

            for i in range(1, 0, -1):
                for tir in range(0, 1, 1):
                    if (len(nextgroup[i]) == i):
                        changeshape(nowshape, nextgroup[i], i)

                        search(searchshow, 0, 999, nowshape, [[0]], [[minarea]], result, lim,
                               i_want_to_save_currentpoint, nextgroup[i], takethepoint, 9999999999)
                        if (len(maxlist) > 0):
                            searchshow = np.array(maxlist[len(maxlist) - 1][0])
                            print('have')
                        pre = len(maxlist)

            if (len(prepremaxlist) != 0):
                collect(prepremaxlist, nowshape)
            if (len(premaxlist) != 0):
                collect(premaxlist, nowshape)
            if (len(preprepremaxlist) != 0):
                collect(preprepremaxlist, nowshape)
            if (len(prepreprepremaxlist) != 0):
                collect(prepreprepremaxlist, nowshape)
            if (len(preprepreprepremaxlist) != 0):
                collect(preprepreprepremaxlist, nowshape)
            if (len(prepreprepreprepremaxlist) != 0):
                collect(prepreprepreprepremaxlist, nowshape)

            if (ggg % 6 == 0 and ggg>0):  # len(maxlist)>len(premaxlist)
                premaxlist = list(maxlist)
            if (ggg % 6 == 1 ):  # and len(maxlist)>len(prepremaxlist)
                prepremaxlist = list(maxlist)
            if (ggg % 6 == 2 ):
                preprepremaxlist = list(maxlist)
            if (ggg % 6 == 3 ):
                prepreprepremaxlist = list(maxlist)
            if (ggg % 6 == 4):
                preprepreprepremaxlist = list(maxlist)
            if (ggg % 6 == 5):
                prepreprepreprepremaxlist = list(maxlist)

            collect(maxlist, nowshape)

            String = 'D:\\multi reg\\'
            String += str(number)
            String += ' '
            String += str(len(maxlist))
            number = number + 1
            String += '.png'
            if (len(maxlist) - 1 > 0):
                searchshow = np.array(maxlist[len(maxlist) - 1][0])

            cv2.imwrite(String, outputimage)
            whitearea=0
            for i in range(0,W,1):
                for j in range(0,W,1):
                    if(outputimage[i][j]>0):
                        whitearea=whitearea+1


            if (len(maxlist) >= countcnt):
                beststring = 'D:\\multi reg\\best '
                beststring += str(len(maxlist))
                countcnt = len(maxlist)
                beststring += ' '
                maxlistforgroup = list(maxlist)
                beststring += str(number)
                beststring += '.png'
                cv2.imwrite(beststring, outputimage)

            if (len(maxlist) > 0):
                retrain()
                net222 = torch.load('net 200 select update with move shoe2 50 out.pkl')

            countmax = 0
            cnt = cnt + 1
            print('ok')
        if (ttt > len(maxlist)):
            ttt = 1

        okk = 0

        print('樟樹=', number)
        print((whitearea/(W*W) )*100)
        if((whitearea/(W*W) )*100>=80 or number==300):
            print(1/0)

        """
        ggg = ggg + 1
        if ggg % 3 == 0:
            if (combo == 5):
                combo = 0

            combo = min(combo + 1, 5)
            searchgroup(0, combo, nowshape, maxlistforgroup, [], [0] * 300, 0, [[1000000]], [])

            ni = nextgroup[combo][0][0]
            nj = nextgroup[combo][0][1]

            for i in range(0, len(nextgroup[combo]), 1):
                nextgroup[combo][i][0] -= ni
                nextgroup[combo][i][1] -= nj
            print(nextgroup)
        """
        ggg = ggg + 1
        ttt = ttt + 1
        if (ggg % 10 == 0):
            lim += 100000