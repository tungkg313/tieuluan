import cv2
import os
import numpy as np
import pandas as pd
import time
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar
from PIL import Image
from imutils import paths


recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

maso =["",]
ten = ['None',] 

def createdata():
    cam = cv2.VideoCapture(0)
    time.sleep(2.0)
    dir = 'database'
    id = stringid.get()
    maso.append(id);
    name = stringname.get()
    ten.append(name)
    f = open('diemdanh.csv','a+')
    f.write(id + ' ; ' + name + '\n')
    f.close()
    path = os.path.join(dir, id)
    if not os.path.isdir(path):
        os.mkdir(path)

    count = 0
    while (True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            count += 1
            pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                if n[0] != '.'] + [0])[-1] + 1
            cv2.imwrite("%s/%s.jpg" % (path, pin) , gray[y:y + h, x:x + w])
            #cv2.imwrite("%s/%s.jpg" % (path, pin) , frame)
        #time.sleep(0.05)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break
        elif count == 100 :
            break
    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Thông Báo", "Tạo dữ liệu của " + name + " thành công^^" + "                                   "
                        + str(count) + " ảnh đã được lưu vào thư mục " +"'"+ id +"'"+ " trong database ")

def resetAction():
    stringid.set("")
    stringname.set("")


def train():
    imagePaths = list(paths.list_images('database'))
    faceSamples=[]
    ids = []
    for (i, imagePath) in enumerate(imagePaths):
        step = "Working on {}".format(imagePath)
        percent["text"] = "{}/{}".format(i + 1, len(imagePaths))
        progress.start()
        status['text'] = "{}".format(step)
        root.update()
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(imagePath.split(os.path.sep)[-2])
        faces = faceCascade.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

def train2():   

    faces,ids = train()
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    messagebox.showinfo("Thông Báo", "Đã thêm dữ liệu sinh viên thành công")
    progress.stop()
    percent["text"] = "Complete"




def nhandien():
    cam = cv2.VideoCapture(0)
    recognizer.read('trainer.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

            id, dubao = recognizer.predict(gray[y:y+h,x:x+w])
            confidence = "  {0}".format(round(100 - dubao))
            if (int(confidence) < 100 and int(confidence) > 40):
                col = ["id","name"]
                diemdanh = pd.read_csv('diemdanh.csv',delimiter=';',names=col)
                id = diemdanh['name'][id-1]

            else:
                id = "unknown"
                count = 0
                for(x,y,w,h) in faces:
                    count += 1
                    #time.sleep(5.0)
                    cv2.imwrite("unknown/Unk." + str(count) + ".jpg", frame)


        
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,0,255), 2)
            cv2.putText(frame, str(confidence)+'%', (x+5,y+h-5), font, 1, (0,0,255), 2)  

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(30) & 0xff
        if key == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def openfile():
    from subprocess import call
    call(["gedit", "diemdanh.csv"])




root = Tk()

root.geometry("375x400")
stringid = StringVar()
stringname = StringVar()
root.title("TLTN/2019 v2.2")
nhan1 = Label(root,text="ỨNG DỤNG ĐIỂM DANH BẰNG NHẬN DIÊN KHUÔN MẶT",fg="red",height=2, anchor=W, relief=GROOVE, bd=2)
frameButton1 = Frame()
Label(frameButton1,text="ID của bạn : ").pack(side=LEFT)
Entry(frameButton1,textvariable=stringid).pack(side=LEFT)
frameButton2 = Frame()
Label(frameButton2,text="Tên của bạn:").pack(side=LEFT)
Entry(frameButton2,textvariable=stringname).pack(side=LEFT)
frameButton3 = Frame()
Button(frameButton3,text="Nhập lại",fg="white",bg="violet",command = resetAction).pack(side=LEFT)
Button(frameButton3,text="Tạo",fg="white",bg="violet",command = createdata).pack(side=LEFT)

nut1 = Button(root,text="Thêm sinh viên/Cập nhật",fg="white",bg="red",width=43,height=2, command=train2)
percent = Label(root, text="         ", anchor=S)
progress = Progressbar(root, length = 370, mode = "determinate")
nut2 = Button(root,text="Điểm Danh/Nhận diện",fg="white",bg="blue",width=43,height=2, command=nhandien)
nut3 = Button(root,text="Kiểm tra dữ liệu",fg="white",bg="green",width=43,height=2, command=openfile)
status = Label(root, text="Nguyễn Sơn Tùng - B1401110", anchor=W, relief=SUNKEN, bd=2)

nhan1.pack(side=TOP)
frameButton1.pack()
frameButton2.pack()
frameButton3.pack()
nut1.pack()
percent.pack()
progress.pack()
nut2.pack()
nut3.pack()
status.pack(side=BOTTOM, fill=X)

root.mainloop()


