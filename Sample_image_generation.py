
import cv2
import numpy as np

img = cv2.imread('35.jpeg')
[row, column] = img.shape[:2]
        
blue,green,red = cv2.split(img)
new_red   = np.zeros([row,column],dtype = 'uint8')
    # color based thresholding
for ia in range(0,row):
    for ja in range(0,column):
        rp = red[ia,ja]
        gp = green[ia,ja]
        bp = blue[ia,ja]
                
        if (rp>100 and gp>100 and bp>100):
               pixel_r = 0 
               pixel_g = 0
               pixel_b = 0
                
        elif ((rp > gp) and (rp > bp)):
               pixel_r = 255
               pixel_g = 0
               pixel_b = 0
                    
        else:
               pixel_r = 0 
               pixel_g = 0
               pixel_b = 0    
            
               
                
new_red[ia,ja]= pixel_r
    
    
    
cv2.imshow('a.jpg',img)
cv2.imshow('c.jpg',new_red)
cv2.waitKey()
cv2.destroyAllWindows()
       
# kernel = np.ones((5,5), np.uint8)
# img_dilation = cv2.dilate(new_red, kernel, iterations=5)
# img_erosion  = cv2.erode(img_dilation, kernel, iterations=3)
   
contours = cv2.findContours(new_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
A = []
R = []
      
for cntr1 in contours:
       
        x,y,w,h = cv2.boundingRect(cntr1)
        area = cv2.contourArea(cntr1)
        A.append(area)
        r = [x,y,x+w,y+h]
        R.append(r)
        A1 = np.array(A)
        R1 = np.array(R)
count = 0    
for i in range(0,len(contours)):
    #(np.max(A))
        if A1[i] <30000:
            a,b,c,d = R1[i]
            new_red[b:d,a:c] = 0

        elif A1[i] > 33000:
            a,b,c,d = R1[i]
            new_red[b:d,a:c] = 0 
        else:
            a1,b1,c1,d1 = R1[i]
            img_crop1 = new_red[b1:d1,a1:c1]
            
            
            
ZW = np.sort(A)
     

cv2.imshow('b.jpg',img_crop1)
cv2.waitKey()
cv2.destroyAllWindows()


