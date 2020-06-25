import cv2
import numpy as np


def show_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
################################################################3
def show_edge(): # canny edge image
    cam = cv2.VideoCapture(0)
    tmax = 200
    tmin = 20

    t=np.random.randint(tmin,tmax)
    t2=int(t/3)
    while True:
        ret_val, img = cam.read()
        edges = cv2.Canny(img, t/3, t)
    #---------------Display-----------------------------------------------------
        cv2.imshow('edge  thresh1 o/p  thesh 2 k/l', edges)
        print('edge  thresh1 o/p t1=' + str(t) + "t k/l t2=" + str(t2))
   #-----------------Control---------------------------------------------------
        ch=cv2.waitKey(1)
        if ch==27: break  # esc to quit
       # ch=chr(ch)p
        print(ch)
        if ch>-1:
            ch=chr(ch)
            if ch=='p': t+=1
            if ch=='o': t-=1
            if ch=='l': t2+=1
            if ch=='k': t2-=1

        if t2>t: t2=t
        t2=np.max([0,t2])
        t = np.max([0, t])
    cv2.destroyAllWindows()
    cam.release()
################################################################3
def show_average(): # average on frames
    cam = cv2.VideoCapture(0)
    rate=30
    ret_val, imgavr = cam.read()
    imgavr=imgavr.astype(np.float64)
    while True:


        ret_val, img = cam.read()
        imgavr=imgavr*(rate-1)/rate+1/rate*img.astype(np.float64)
        cv2.imshow('my average  o/p', cv2.resize(imgavr.astype(np.uint8),(1536,864)))
        print(rate)
        # -----------------Control---------------------------------------------------
        ch = cv2.waitKey(1)
        if ch == 27: break  # esc to quit
        # ch=chr(ch)p
        print(ch)
        if ch > -1:
            ch = chr(ch)
            if ch == 'p': rate += 1
            if ch == 'o': rate -= 1

        rate = np.max([0, rate])
    cv2.destroyAllWindows()
    cam.release()
################################################################3
def show_difAvg(): #difference bwteeen frames time delay
        cam = cv2.VideoCapture(0)
        rate = float(np.random.randint(5, 29))
        ret_val, img = cam.read()
        ret_val, img2 = cam.read()
        imgavr = np.abs(img-img2).astype(np.float64)
        while True:

            ret_val, img = cam.read()
            cv2.waitKey(5)
            ret_val, img2 = cam.read()
            img=np.abs(img-img2)
            imgavr = imgavr * (rate - 1) / rate + 1 / rate * img.astype(np.float64)
            cv2.imshow('my webcam o/p', cv2.resize(imgavr.astype(np.uint8),(1536,864)))
            # -----------------Control---------------------------------------------------
            ch = cv2.waitKey(1)
            if ch == 27: break  # esc to quit
            # ch=chr(ch)p
          #  print(ch)
            if ch > -1:
                ch = chr(ch)
                if ch == 'p': rate += 1
                if ch == 'o': rate -= 1
            if rate<0: rate=0
        cv2.destroyAllWindows()
        cam.release()
################################################################3
def show_dif(): # difference between frame not good
        cam = cv2.VideoCapture(0)
   #     rate = float(np.random.randint(5, 29))
        rate = 5#float(np.random.randint(5,20))
        thresh=50
        ret_val, imgavr = cam.read()
        while True:

            ret_val, img = cam.read()
            ret_val, img2 = cam.read()
            img2 = imgavr * (rate - 1) / rate + 1 / rate * img2.astype(np.float64)
            dif=(np.abs(np.mean((img-img2),2))>(thresh)).astype(np.uint8)
            img2[:,:,0]*=dif
            img2[:, :, 1] *= dif
            img2[:, :, 2] *= dif
            imgavr = imgavr * (rate - 1) / rate + 1 / rate * img2.astype(np.float64)

            cv2.imshow('my webcam o/p k/l', cv2.resize(imgavr.astype(np.uint8),(1536,864)))
            # -----------------Control---------------------------------------------------
            ch = cv2.waitKey(1)
            if ch == 27: break  # esc to quit
            # ch=chr(ch)p
            #  print(ch)
            if ch > -1:
                ch = chr(ch)
                if ch == 'p': rate += 1
                if ch == 'o': rate -= 1
                if ch == 'l': thresh += 1
                if ch == 'k': thresh -= 1
            if rate < 0: rate = 0
            if thresh<0: thresh=0
        cv2.destroyAllWindows()
        cam.release()
##############################Psychedlic#################################################################
def Psychedlic(): # map color using none linear function
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.2,1,3],dtype=np.float) # Rate of change (no sign)
            dif=0.1
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][img[:,:,i]].astype(np.uint8)

                    cv2.imshow('e/r d/f c/v', cv2.resize(DispImg,(1536,864)))
                    print(d)
                    print((sgn==-1).sum(1))
                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(1)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def Psychedlic2(): # map color using none linear function
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.1,0.05,1],dtype=np.float) # Rate of change (no sign)
            dif=0.1
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    # sgn[lut > 255] =-1
                    # sgn[lut < 0] = 1
                    lut[lut > 255] = 0
                    lut[lut < 0] = 255
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][img[:,:,i]].astype(np.uint8)

                    cv2.imshow('e/r d/f c/v', cv2.resize(DispImg,(1536,864)))
                    print(d)
                    print((sgn==-1).sum(1))
                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(1)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def Psychedlic3(): # good map color using none linear function
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.5,0.1,1],dtype=np.float) # Rate of change (no sign)
            dif=0.1
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]*(lut[i]+1)/20
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][img[:,:,i]].astype(np.uint8)

                    cv2.imshow('e/r d/f c/v', cv2.resize(DispImg,(1536,864)))
                    print(d)
                    print((sgn==-1).sum(1))
                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(1)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def Psychedlic4(): # good map color using none linear function
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.3,0.05,0.6],dtype=np.float) # Rate of change (no sign)
            dif=0.05
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]*(lut[i]+1)/20
                    # sgn[lut > 255] =-1
                    # sgn[lut < 0] = 1
                    lut[lut > 255] = 0
                    lut[lut < 0] = 255
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][img[:,:,i]].astype(np.uint8)

                    DispImg=cv2.GaussianBlur(DispImg, (3, 3), cv2.BORDER_DEFAULT)

                    cv2.imshow('e/r d/f c/v',  cv2.resize(DispImg,(1536,864)))
                    print(d)
                    print((sgn==-1).sum(1))
                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(1)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def PsychedlicDif(): #  good
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.1,0.5,1.5],dtype=np.float) # Rate of change (no sign)
            dif=0.1
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            ret_val, Previmg = cam.read()
            AvDisp=Previmg.copy()
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()

                    difImg=np.abs(img - Previmg)
                   # Previmg = img.copy()
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][difImg[:,:,i]].astype(np.uint8)
                    DispImg = cv2.GaussianBlur(DispImg, (3, 3), cv2.BORDER_DEFAULT)
                    Previmg=DispImg.copy()
                    AvDisp=(AvDisp*0.75+Previmg*0.25).astype(np.uint8)
                    cv2.imshow('e/r d/f c/v', cv2.resize(AvDisp,(1536,864)))#DispImg)
                    print(d)

                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(10)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def PsychedlicCool(): # Great
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.2 , 0.45, 0.85],dtype=np.float) # Rate of change (no sign)
            dif=0.05
            Fract=0.2
            Dfract=0.02
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            ret_val, Previmg = cam.read()
            AvFrame=Previmg.copy()
            PrevFrame = Previmg.copy()
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()

                    difImg=np.abs(img - Previmg)
                   # Previmg = img.copy()
                    AvFrame = ((difImg * ((1000.0 - Fract) / 1000.0) + AvFrame * (Fract / 1000.0)))
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][AvFrame[:,:,i].astype(np.uint8)].astype(np.uint8)
                    DispImg = cv2.GaussianBlur(DispImg, (3, 3), cv2.BORDER_DEFAULT)

                    Previmg=DispImg.copy()
                    PrevFrame = (PrevFrame * 0.7 + 0.3 * DispImg.copy()).astype(np.uint8)
                    cv2.imshow('fract o,p dif e/r d/f c/v', cv2.resize(PrevFrame,(1536,864)))#DispImg)

                    print(d)
                    print(Fract)

                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(10)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                        if ch == 'o': Fract += Dfract
                        if ch == 'p': Fract -= Dfract
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def PsychedlicCool2(): # Great
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.2 , 0.45, 0.85],dtype=np.float) # Rate of change (no sign)
            dif=0.05
            Fract=0.2
            Dfract=0.02
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            ret_val, Previmg = cam.read()
            AvFrame=Previmg.copy()
            PrevFrame=Previmg.copy()
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, img = cam.read()
                    ##--------------------------------
                    DispImg=img.copy()

                    difImg=np.abs(img - Previmg)
                   # Previmg = img.copy()
                    AvFrame = ((difImg * ((1000.0 - Fract) / 1000.0) + AvFrame * (Fract / 1000.0)))
                    DispImg = cv2.GaussianBlur(DispImg, (3, 3), cv2.BORDER_DEFAULT)
                    for i in range(3):
                          DispImg[:,:,i]=lut[i][difImg[:,:,i].astype(np.uint8)].astype(np.uint8)

                    Previmg=DispImg.copy()
                    PrevFrame= (PrevFrame*0.7+DispImg*0.3).astype(np.uint8)
                    cv2.imshow('fract o,p dif e/r d/f c/v', cv2.resize(PrevFrame,(1536,864)))
                    print(d)
                    print(Fract)

                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(10)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                        if ch == 'o': Fract += Dfract
                        if ch == 'p': Fract -= Dfract
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
##############################Psychedlic#################################################################
def PsychedlicStrange3(): #
            cam = cv2.VideoCapture(0)
            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0,2 , 0.4,  2],dtype=np.float) # Rate of change (no sign)
            dif=0.1
            Fract=0.3
            Dfract=0.02
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            ret_val, Previmg = cam.read()
            AvFrame=Previmg.copy()
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, Frame = cam.read()
                    ##--------------------------------
                    PrevFrame = Frame.copy()
                    AvFrame = ((Frame * ((1000.0 - Fract) / 1000.0) + AvFrame * (Fract / 1000.0)))
                    OldAvFrame = AvFrame.copy();
                    AvFrame = cv2.GaussianBlur(AvFrame, (3, 3), cv2.BORDER_DEFAULT)



                    for i in range(3):
                        AvFrame[:,:,i]=lut[i][AvFrame[:,:,i].astype(np.uint8)].astype(np.uint8)
                    cv2.imshow('fract o,p dif e/r d/f c/v', AvFrame.astype(np.uint8))
                    AvFrame=OldAvFrame
                    print(d)
                    print(Fract)

                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(10)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                        if ch == 'o': Fract += Dfract
                        if ch == 'p': Fract -= Dfract
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()

##############################Psychedlic#################################################################
def PsychedlicStrange2(): #
            cam = cv2.VideoCapture(0)

            #--Look up tables for colors any color in the image will be map in the corresponding color here
            lut=np.ones([3,256],dtype=np.float) # color look up table
            sgn = np.ones([3,256], dtype=np.float) # color direction of change
            d=np.array([0.1 , 0.5,  2],dtype=np.float) # Rate of change (no sign)
            dif=0.05
            Fract=0.3
            Dfract=0.02
            for i in range(256):
                lut[:,i]=[i,i,i]

            #-------Main loop--------------------------------------
            ret_val, PrevFrame = cam.read()
            AvFrame=PrevFrame.copy()
            AvDisp=PrevFrame.copy()
            while True:
                    # -------Update look up table value
                    for i in range(3): lut[i]+=sgn[i]*d[i]
                    sgn[lut > 255] =-1
                    sgn[lut < 0] = 1
                    lut[lut > 255] = 255
                    lut[lut < 0] = 0
                    ##-------get image----------------
                    ret_val, Frame = cam.read()
                    ##--------------------------------
                    DifFrame = np.abs(Frame - PrevFrame)

                    PrevFrame = Frame.copy();
                    AvFrame = ((DifFrame * ((1000.0 - Fract) / 1000.0) + AvFrame * (Fract / 1000.0)));
                    OldAvFrame = AvFrame.copy();
                    AvFrame = cv2.GaussianBlur(AvFrame, (3, 3), cv2.BORDER_DEFAULT)



                    for i in range(3):
                        AvFrame[:,:,i]=lut[i][AvFrame[:,:,i].astype(np.uint8)].astype(np.uint8)
                    AvDisp=(AvFrame*0.1+AvDisp*0.9)
                    cv2.imshow('fract o,p dif e/r d/f c/v', AvDisp.astype(np.uint8))
                    AvFrame=OldAvFrame
                    print(d)
                    print(Fract)

                    ##-----------------Control---------------------------------------------------
                    ch = cv2.waitKey(10)
                    if ch == 27: break  # esc to quit
                    # ch=chr(ch)p
                    #  print(ch)
                    if ch > -1:
                        ch = chr(ch)
                        if ch == 'e': d[2] += dif
                        if ch == 'r': d[2] -= dif
                        if ch == 'd': d[1] += dif
                        if ch == 'f': d[1] -= dif
                        if ch == 'c': d[0] += dif
                        if ch == 'v': d[0] -= dif
                        if ch == 'o': Fract += Dfract
                        if ch == 'p': Fract -= Dfract
                    d[d<0]=0
            cv2.destroyAllWindows()
            cam.release()
#PsychedlicStrange3()
#PsychedlicStrange2()
#PsychedlicDif()
while(True):
    PsychedlicDif()
    PsychedlicCool()
    #PsychedlicDif()
   #
   # # PsychedlicCool2()
   #  Psychedlic3()
   #  Psychedlic4()
   #  show_average()

