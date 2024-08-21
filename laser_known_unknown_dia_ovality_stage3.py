'''
https://blog.roboflow.com/computer-vision-measure-distance/
https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
https://www.cmrp.com/ovalitycalc-php-template
relation between numpy and opencv
1. x and y coordinates are interchange
2. rgb channels are interchange

Version-3 combination of first and second model

'''
# video to frame
import cv2

vid = cv2.VideoCapture('C:/Users/User/Desktop/laser_field/20240529_215416_1_fold/20240529_215416_1.avi')
sucess,frame = vid.read()
count = 0


while sucess:
    count+=1
    suce, fname = vid.read()
    if suce == False:
        break
    cv2.imwrite('image_%05d.jpg' % count,fname)
###############################################################################
# import the opencv library
# for live video 
import cv2 


# define a video capture object 
vid = cv2.VideoCapture(0) 

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
    
###############################################################################

# design the cheker board
import cv2
import numpy as np
from matplotlib import pyplot as plt

black = np.zeros([200,200],dtype = 'uint8')
white = (np.ones([200,200],dtype = 'uint8'))*255

#[rows, col] = A.shape

k = 0
k1 = 0
r = 1200
merged_img = []
merged_img1 = []

# odd rows
for i in range(0,r,200):
    if k%2 == 0:
       G = white
    else:
       G = black
    merged_img.append(G)
    k = k+1

result_h = np.hstack(merged_img)

# even rows
for i1 in range(0,r,200):
    if k1%2 == 0:
       G1 = black
    else:
       G1 = white
    merged_img1.append(G1)
    k1 = k1+1

result_h1 = np.hstack(merged_img1)

verticall = []
for u in range (0,5):
    if u%2 == 0:
        F = result_h1
    else:
        F = result_h
    verticall.append(F)


result_v = np.vstack(verticall)



#cv2.imshow('A.jpg',result_h)
#cv2.imshow('A1.jpg',result_h1)
cv2.imshow('A1.jpg',result_v)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("p_checkerboad3.png",result_v)   

###############################################################################
## detecting the checkerboard edge

import cv2

# Load the image
img = cv2.imread('F:/23.02.2024/240223-002/14.jpeg')
#img = cv2.GaussianBlur(org_img, (25,25), 21)
#img = cv2.resize(img1,[1024,720])

# Define the number of rows and columns in the chessboard
n_rows = 6
n_cols = 8

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the corners of the chessboard
ret, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), None)

# Refine the corners to subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)

# Draw the corners on the image
cv2.drawChessboardCorners(img, (n_rows, n_cols), corners, ret)

gh = corners.reshape(len(corners),2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
###############################################################################

# import cv2
# import numpy as np
# import os
# import glob
# from scipy.spatial import distance as dist 
# # Defining the dimensions of checkerboard
# CHECKERBOARD = (6,8)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Creating vector to store vectors of 3D points for each checkerboard image
# objpoints = []
# # Creating vector to store vectors of 2D points for each checkerboard image
# imgpoints = [] 
# focal = []
# #distance = 500
# actual_length = 26
# # Defining the world coordinates for 3D points
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# #objp = objp * 26
# #prev_img_shape = None

# # Extracting path of individual image stored in a given directory
# images = glob.glob('D:/laser profiling/240222-003_mds_testing/New folder/*.jpeg')
# for fname in images:
#     #img = cv2.imread(fname)
#     img = cv2.imread('D:/laser profiling/240222-003_mds_testing/New folder/3.jpeg')
#     #img = cv2.resize(img,[1280,720],interpolation = cv2.INTER_CUBIC)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     # If desired number of corners are found in the image then ret = true
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
#     	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
#     """
#     If desired number of corner are detected,
#     we refine the pixel coordinates and display 
#     them on the images of checker board
#     """
#     if ret == True:
#         objpoints.append(objp)
#         # refining pixel coordinates for given 2d points.
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
#     cv2.imshow('img',img)
#     cv2.waitKey(0)

#     NCC1 = np.reshape(corners,[48,2])

#     NC1 = NCC1.copy()
  
#     NC1[:,0] = NCC1[:,1]
#     NC1[:,1] = NCC1[:,0]
#     h,w = img.shape[:2]
    
    
#     d1 = NC1[1,0]-NC1[0,0]
#     d2 = NC1[2,0]-NC1[1,0]
#     d3 = NC1[3,0]-NC1[2,0]
#     d4 = NC1[4,0]-NC1[3,0]
#     d5 = NC1[5,0]-NC1[4,0]
    
#     d = (d1+d2+d3+d4+d5)/5

#     d05 = NC1[5,0]-NC1[0,0] 
    
# cv2.destroyAllWindows()




###############################################################################

# https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse
#start 

import cv2
import numpy as np
from scipy.spatial import distance as dist 
from os import listdir 
from matplotlib import pyplot as plt
import circle_fit as cf
#import math

input_path = 'C:/Users/User/Desktop/laser_field/20240529_215416_1_fold/original_image/'
#input_path = 'D:/laser profiling/Result_26122023/case4/original_image_296mm/'
output_path_rgb = 'C:/Users/User/Desktop/laser_field/20240529_215416_1_fold/processed_rgb/'
output_path_bw = 'C:/Users/User/Desktop/laser_field/20240529_215416_1_fold/processed_bw/'


diameter_known = 1

img_list = listdir(input_path)
PPM = None

orig_dia_mm = 152.4

R_pix =[] # radius in pixels
D_mm = []
Ovality_all = []
Csl_rad = []


#MMP = 26/53.2 ## first col average, 26 mm is actual length of checkerbox- img3
#MMP = (26/53.2628) # first block - img3
#MMP = (26/53.299)



#ind = 1
for ind in range(0,len(img_list)):
    
    combine_rad = []
    sub_rad_dis = []
    mul_dia_mm  = []

    mul_rad_pix = []
    multiple_xc = []
    multiple_yc = []
    
    #img = cv2.imread('D:/laser profiling/Result_26122023/case4/original_image_296mm/image_00099.jpg')
    img = cv2.imread(input_path + img_list[ind])

    [row, column] = img.shape[:2]
    
    
    blue,green,red = cv2.split(img)
    new_red   = np.zeros([row,column],dtype = 'uint8')
    # new_green = np.zeros([row,column],dtype = 'uint8')
    # new_blue  = np.zeros([row,column],dtype = 'uint8')
    # color based thresholding
    for ia in range(0,row):
        for ja in range(0,column):
            rp = red[ia,ja]
            gp = green[ia,ja]
            bp = blue[ia,ja]
            
            if (rp>200 and gp>200 and bp>200):
                pixel_r = 0 
                pixel_g = 0
                pixel_b = 0
            elif ((rp>245) and (gp<255 and bp<255)):
                pixel_r = 255 
                pixel_g = 0
                pixel_b = 0
            else:
                pixel_r = 0 
                pixel_g = 0
                pixel_b = 0
           
            
            new_red[ia,ja]= pixel_r
            # new_green[i,j]= pixel_g
            # new_blue[i,j]= pixel_b


    # cv2.imshow('a.jpg',img)
    # #cv2.imshow('b.jpg',new_red)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    #newRGBImage = cv2.merge((new_blue,new_green,new_red))
    
    #yx_coords1 = np.column_stack(np.where(new_red == 255))
    # xy_coords = yx_coords.copy()

    # xy_coords[:,0] = yx_coords[:,1]
    # xy_coords[:,1] = yx_coords[:,0]
    
    xy_coords2 = cv2.findNonZero(new_red)    # finding the edge pixels coordinates
    xy_coords  = xy_coords2.reshape(len(xy_coords2),2)
    
    '''
    A = np.array([1,0,5,8,0,0,4,7])
    B = cv2.findNonZero(A)
    bb = len(B)
    C = B.reshape(bb,2)
    '''
    xc,yc,r,_ = cf.least_squares_circle(xy_coords)
    for ib in range(0,len(xy_coords)):
        sb = xy_coords[ib]
        calcu_rad = int(dist.euclidean((xc,yc),(sb[0],sb[1])))
        combine_rad.append(calcu_rad)
        #cv2.circle(img,(sb[0],sb[1]),2,(100,255,150),-1) # original laser circle coordinates
        
    
    #R.append(r) # radius in pixels
    if diameter_known == 1:
        
        if PPM == None:
            PPM = (2*r)/(orig_dia_mm)
            Rad = (2*r/PPM)
          
        else:
            Rad = ((2*r)/PPM) 
          
        
        #R_pix.append(r) # radius in pixels
        D_mm.append(Rad)
    
    else:
        
        Rad = MMP *2*r
        #Csl_rad.append(Rad)
    
    
    cv2.circle(img,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
    #cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1)  # fit circle radius
    #cv2.circle(img,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
    
    cv2.circle(new_red,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
    cv2.circle(img,(int(xc),int(yc)),5,(0,0,255),-1)  # fit circle radius
    #cv2.circle(new_red,(int(column/2),int(row/2)),5,(0,255,255),-1)  # fit circle radius
    
    
    # ovality calculation
    
    # CR = np.array(combine_rad) 
    # all_measure_radius = np.sort(CR)
    # # calculate distance between center and the each edge point; if the distance is more than or less than 10 then remove those points 
    # index_end = np.where(all_measure_radius>((r+1))) 
    # index_start = np.where(all_measure_radius<(r-1))
    
    # new_measure_radius = np.delete(all_measure_radius, index_end)  
    # new_measure_radius = np.delete(new_measure_radius, index_start) # deleting all calculated radius value which is out of index 
    '''
    A = np.array([1,2,5,9,8,3,7])
    
    B = np.sort(A)
    
    index_end = np.where(B>8)
    index_start = np.where(B<2)
    
    C = B.copy()
    
    new_B = np.delete(B, index_end)
    new_B = np.delete(new_B, index_start) 
    '''
    #ovality_cal = ((new_measure_radius[-1] - new_measure_radius[0]) /r)*100
    #Ovality_all.append(ovality_cal)
    cv2.putText(img, "{:.1f}mm".format(Rad),(500,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (0, 0, 255), 4)
    #cv2.putText(img, "{:.1f}percentage".format(ovality_cal),(100,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (0, 100, 255), 4)
    
    cv2.putText(new_red, "{:.1f}mm".format(Rad),(500,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4)
    #cv2.putText(new_red, "{:.1f}percentage".format(ovality_cal),(100,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 255, 255), 4)
    
    cv2.imwrite(output_path_rgb + img_list[ind],img)
    cv2.imwrite(output_path_bw + img_list[ind],new_red)
    
    # cv2.imshow('a1.jpg',img)
    # cv2.imshow('b1.jpg',new_red)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    


# Cslrad = np.array(Csl_rad)    

# np.save('Calculate_dia.npy',Cslrad)
# length = len(Cslrad)
# call_diaa = np.mean(Cslrad) 
# q_101 = np.repeat(call_diaa,length)  



#np.save('pixel_radius.npy',R_pix)
np.save('Dia_mm.npy',D_mm)
np.save('ovality_percentage',Ovality_all)   

    
Mean_dia = np.mean(D_mm)
Oval1 = np.array(Ovality_all)   
#R1 = np.array(R_pix)
D1 = np.array(D_mm)
#mean_rad_pix = np.mean(D1)
length = len(D1)
q_100 = list(range(0,length))
#mean_cal_diameter = np.repeat(Mean_dia,length)
mean_cal_diameter = np.repeat(152.4,length)
#mean_cal_rad = np.repeat(mean_rad_pix,length)  

  
fig = plt.figure(figsize=(10,5))
plt.ylim(100,200)
plt.plot(q_100, D1,'black') 
#plt.plot(q_100, Cslrad,'black') 
plt.plot(q_100, mean_cal_diameter,'red')
#plt.plot(q_100, Oval1,'black') 
#plt.plot(q_100,D_mm,'red')
#plt.plot(q_100,mean_cal_diameter,'blue')
#plt.plot(q_100,mean_cal_rad,'red')
plt.xlabel('pipe length') 
plt.ylabel('Diameter') 
fig.savefig('dia_pix1.jpg', bbox_inches='tight', dpi=150)
plt.show()
    



# #plt.ylim(250,350)
# plt.plot(q_100, Calrad,'black') 
# plt.plot(q_100, q_101,'red')
# plt.xlabel('pipe length') 
# plt.ylabel('Calculated Diameter')
# plt.title('Diameter calculation with PTZ camera') 
# plt.grid(True)
# fig.savefig('dia1_300mm_.jpg', bbox_inches='tight', dpi=150)
# plt.show()
    
    
   
   