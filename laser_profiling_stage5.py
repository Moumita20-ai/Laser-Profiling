
'''
The fundamental matrix relates corresponding points between a pair of
    uncalibrated images. The matrix transforms homogeneous image points in one
    image to epipolar lines in the other image.

    The fundamental matrix is only defined for a pair of moving images. In the
    case of pure rotation or planar scenes, the homography describes the
    geometric relation between two images (`ProjectiveTransform`). If the
    intrinsic calibration of the images is known, the essential matrix describes
    the metric relation between the two images (`EssentialMatrixTransform`).
'''

import cv2
import numpy as np
from scipy.spatial import distance as dist 
from os import listdir 
from matplotlib import pyplot as plt
import circle_fit as cf
from skimage import transform

input_path = 'C:/Users/User/Desktop/New folder_2902/small_wheel/mds_290/original_image/'

output_img_path = 'C:/Users/User/Desktop/New folder_2902/small_wheel/mds_290/with_homography/processed_image_250/'


img_list = listdir(input_path)
img_size = None

PPM = 1.364864864864865  #### (202/148)


R_pix =[] # radius in pixels
D_mm = []
Ovality_all = []
Cal_rad = []
#ind = 1
Rratio = []
for ind in range(0,len(img_list)):
    
    combine_rad = []
    sub_rad_dis = []
    mul_dia_mm  = []
    
    mul_rad_pix = []
    multiple_xc = []
    multiple_yc = []

    img = cv2.imread(input_path + img_list[ind])
    [row, column] = img.shape[:2]
        
        
    blue,green,red = cv2.split(img)
    new_red   = np.zeros([row,column],dtype = 'uint8')
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
                elif ((rp>220) and (gp<200 and bp<200)):
                    pixel_r = 255 
                    pixel_g = 0
                    pixel_b = 0
                else:
                    pixel_r = 0 
                    pixel_g = 0
                    pixel_b = 0
               
                
                new_red[ia,ja]= pixel_r
    
    
    xy_coords2 = cv2.findNonZero(new_red)    # finding the edge pixels coordinates
    xy_coords  = xy_coords2.reshape(len(xy_coords2),2)
                                    
    xc,yc,r,_ = cf.least_squares_circle(xy_coords)
                
    for ib in range(0,len(xy_coords)):
            sb = xy_coords[ib]
            calcu_rad = int(dist.euclidean((xc,yc),(sb[0],sb[1])))
            combine_rad.append(calcu_rad)
            cv2.circle(img,(sb[0],sb[1]),1,(100,255,150),-1) # original laser circle coordinates
                
    cv2.circle(img,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
                
    sample_img = np.zeros([720,1280],dtype = 'uint8')
    cv2.circle(sample_img,(int(xc),int(yc)),int(r),(255,0,0),2) # fit circle
    contours1 = cv2.findContours(sample_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]        
    for cntr11 in contours1:
            #orig = new_red.copy()
            x1,y1,w1,h1 = cv2.boundingRect(cntr11)            
           
    img_crop1 = sample_img[y1:y1+h1,x1:x1+w1]
    
    if (img_size == None):
        img_size = img_crop1.shape[0]
        
    img_crop = cv2.resize(img_crop1,[img_size,img_size],interpolation = cv2.INTER_CUBIC)
   
    
    source_img = img_crop.copy()
    
    dst_img = plt.imread('background_removal_crop.jpg')
    S_shape = source_img.shape[0]
    D_shape = int((dst_img.shape[0] + dst_img.shape[1])/2)
    [y_d,x_d] = dst_img.shape

    
    ratio =  round(S_shape/  D_shape , 2)
    Rratio.append(ratio)

    
    dst_2 = np.array([0,0,
                      int(x_d/2),0,
                      x_d,0,
                      x_d,int(y_d/2), 
                      x_d,y_d,
                      int(x_d/2),y_d, 
                      0,y_d,
                      0,int(x_d/2),]).reshape((8, 2))

    
    
    src_2 = np.array([0,0,
                      (int(w1/2)),0,
                      w1,0,
                      w1,(int(h1/2)), 
                      w1,h1,
                      (int(w1/2)),h1, 
                      0,h1,
                      0,(int(h1/2)),]).reshape((8, 2))
    

    
    
    # fig, ax = plt.subplots(3, 1, figsize=(25, 15))
    # ax[0].grid(True)
    # ax[0].imshow(source_img, )
    # #fig.savefig('pipe_400_source.jpg')
    # ax[0].scatter(src_2[:,0], src_2[:,1], c='red', s=30)
    # ax[0].set_title('source coordinates')

    # ax[1].grid(True)
    # ax[1].imshow(dst_img)
    # #fig.savefig('pipe_150_destination.jpg')
    # ax[1].scatter(dst_2[:,0], dst_2[:,1], c='red', s=30)
    # ax[1].set_title('destination coordinates')

    dst_2 = dst_2*(ratio) #because image sizes are not the same.
    #dst_2 = dst_2*(2.075) #because image sizes are not the same. 2.05
    tform = transform.estimate_transform('projective', src_2, dst_2)
    tf_img = transform.warp(source_img, tform.inverse)

    # ax[2].grid(True)
    # ax[2].imshow(tf_img)
    # #fig.savefig('pipe_400_destination_homography.jpg')
    # ax[2].scatter(dst_2[:,0], dst_2[:,1], c='red', s=10)
      
    circle_cord = cv2.findNonZero(tf_img)    # finding the edge pixels coordinates
    circle_cord_reshape  = circle_cord.reshape(len(circle_cord),2)
                        
    c_x,c_y,rad,_ = cf.least_squares_circle(circle_cord_reshape)
                            
    radius_mm = ((2*rad)/PPM)
                    
    Cal_rad.append(radius_mm)    
    
    # ovality calculation
                        
    CR = np.array(combine_rad) 
    all_measure_radius = np.sort(CR)
    # #calculate distance between center and the each edge point; if the distance is more than or less than 10 then remove those points 
    index_end = np.where(all_measure_radius>((r+10))) 
    index_start = np.where(all_measure_radius<(r-10))
                        
    new_measure_radius = np.delete(all_measure_radius, index_end)  
    new_measure_radius = np.delete(new_measure_radius, index_start) # deleting all calculated radius value which is out of index 
                     
    ovality_cal = ((new_measure_radius[-1] - new_measure_radius[0]) /r)*100
    # #inserting text in the original image
    Ovality_all.append(ovality_cal)

    
    tf_img1 = tf_img*255
    tf_img2 = tf_img1.astype(np.uint8)
    
    cv2.putText(img, "{:.1f}percentage".format(ovality_cal),(100,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (0, 100, 255), 4)

    cv2.putText(img, "{:.1f}mm".format(radius_mm),(500,500), cv2.FONT_HERSHEY_SIMPLEX,0.85, (255, 0, 255), 4)                   
    

    cv2.imwrite(output_img_path + img_list[ind],img)                    
    
    
    # cv2.imshow('d.jpg',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    #print(f"PPM is:{PPM}")


Calrad = np.array(Cal_rad)    

## comparison with standard diameter range : https://www.octalsteel.com/steel-pipe-dimensions-sizes/
mean_dia = np.mean(Calrad)
 
stand_dia = np.array([250,300,350,400,450,500,600,700])

sub_dia = abs(stand_dia - mean_dia)

P_diameter = stand_dia[np.argmin(sub_dia)]

print(f'The final pipe diameter is: {P_diameter}')

np.save('Calculate_dia.npy',Calrad)

length = len(Calrad)
q_100 = list(range(0,length))


call_diaa = np.mean(Calrad) 
q_101 = np.repeat(call_diaa,length)  

 
fig = plt.figure(figsize=(10,5))
#plt.ylim(250,350)
plt.plot(q_100, Calrad,'black') 
plt.plot(q_100, q_101,'red')
plt.xlabel('pipe length') 
plt.ylabel('Calculated Diameter')
plt.title('Diameter calculation with PTZ camera') 
plt.grid(True)
fig.savefig('dia1_300mm_.jpg', bbox_inches='tight', dpi=150)
plt.show()
    

