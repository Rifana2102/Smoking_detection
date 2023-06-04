### IMPORT NECESSARY LIBRARIES
from ultralytics import YOLO
import numpy as np
import pandas as pd
import math
import time
import cv2

### FILENAME OF THE VIDEO (INSIDE THE "input" folder)
fname = 'vid.mp4'

### DEVICE TO USE <'cpu', '0', or 'mps'>
device = '0'

showImg_flag = True
saveImg_flag = True
blur_flag = True
mask_conf_decay = 0.00025

### EDIT THESE POINTS BASED ON THE ACTUAL 2D AND 3D TRANSLATION POINTS
image_points_2D = np.array([
                            (90, 415),          # point A
                            (680, 310),         # point B
                            (155, 680),         # point C
                            (1190, 500),        # point D
                            (1805, 620),        # point E
                            (239, 993),         # point F
                        ], dtype="double")

figure_points_3D = np.array([
                            (0.3, 0.0, 0.0),    # A in meters
                            (3.9, 0.0, 0.0),    # B in meters
                            (0.3, 2.7, 0.0),    # C in meters
                            (4.8, 3.3, 0.0),    # D in meters
                            (5.7, 6.0, 0.0),    # E in meters
                            (0.3, 3.6, 0.0),    # F in meters
                        ])   


### SETTING UP VARIABLES FOR VECTOR ROTATION AND TRANSLATION
vid_cap = cv2.VideoCapture('input/'+fname)
fps = vid_cap.get(cv2.CAP_PROP_FPS)
width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_cap.release()
print('height:'+str(height)+' width:'+str(width))
distortion_coeffs = np.zeros((4,1))
focal_length = width
center = (width/2, height/2)
matrix_camera = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )
success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0)

#prepare rotation matrix
R_mtx, jac=cv2.Rodrigues(vector_rotation)

#prepare projection matrix 
Extrincic=cv2.hconcat([R_mtx,vector_translation])
Projection_mtx=matrix_camera.dot(Extrincic)

#delete the third column since Z=0 
Projection_mtx = np.delete(Projection_mtx, 2, 1)

#finding the inverse of the matrix 
Inv_Projection = np.linalg.inv(Projection_mtx)

### rect = [x1,y1,x2,y2]
def get_world_xy(rect):
    # detected image point
    img_point = np.array(((rect[0]+rect[2])/2.0, rect[3]))
    # adding dimension 1 in order for the math to be correct (homogeneous coordinates)
    img_point=np.hstack((img_point,np.array(1)))
    # calculating the 3D point which located on the 3D plane
    three_D_point=Inv_Projection.dot(img_point)
    # get the converted values
    x_world = three_D_point[0]/three_D_point[2]
    y_world = three_D_point[1]/three_D_point[2]
    return x_world,y_world


def pixelate_face(image, block_size=10):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, int(w/block_size) + 1, dtype="int")
	ySteps = np.linspace(0, h, int(h/block_size) + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image

### RETURNS THE AREA OF INTERSECTION WHERE rect_* = [x1,y1,x2,y2]
def area_intersection(rect_a,rect_b):
    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[2], rect_b[2])
    y2 = min(rect_a[3], rect_b[3])
    if (x2-x1)>0 and (y2-y1)>0:
        return (x2-x1)*(y2-y1)
    else:
        return 0
    
def make_square_based_on_width(rect_a):
    return  [
                rect_a[0],
                rect_a[1],
                rect_a[2],
                min(
                    rect_a[3],
                    max(
                        rect_a[1]+(rect_a[2]-rect_a[0]),
                        rect_a[1]+int((rect_a[3]-rect_a[1])*0.5)
                        )
                    )
            ]

t1 = time.time()

### MODEL FOR PERSON TRACKING; INITIALIZE MODEL FOR STREAMING 
model = YOLO('models/yolov8x.pt')
frame_results = model.track(source='input/'+fname, classes=0, stream=True, device=device)

### MODEL FOR MASK DETECTION (CUSTOM TRAINED MODEL)
mask_model = YOLO('models/mask_YOLOv8_xLarge.pt')

### mask_dict = {id:[mask_status, mask_conf]}
mask_dict = {}
time_frame = 0
csv_timer = 0

df = pd.DataFrame(columns=['frame','person','bbox_xmin','bbox_xmax','bbox_ymin','bbox_ymax','mask_stat','mask_conf','x_world','y_world'])

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
color = (255, 255, 255)
bg_color = (100,0,0)
thickness = 1
lineType = cv2.LINE_AA

### Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out_tracked = cv2.VideoWriter('output/tracked_'+fname, fourcc, fps, (int(width),int(height)))
vid_out_blurred = cv2.VideoWriter('output/blurred_'+fname, fourcc, fps, (int(width),int(height)))

### MAIN LOOP
for result in frame_results:
    boxes = result.boxes
    frame = result.orig_img
    frame_copy = frame.copy()

    if len(boxes)>0:
        ### DETECT MASK FROM FACES
        mask_results = mask_model.predict(frame, device=device, verbose=False)
        ### BLUR THE FACES DETECTED
        if blur_flag and len(mask_results) > 0:
            for box_face in mask_results[0].boxes:
                x1tmp, y1tmp, x2tmp, y2tmp = np.squeeze(box_face.xyxy.tolist())
                x1tmp, y1tmp, x2tmp, y2tmp = int(x1tmp), int(y1tmp), int(x2tmp), int(y2tmp)
                crop_obj_face = frame[y1tmp:y2tmp,x1tmp:x2tmp]
                frame[y1tmp:y2tmp,x1tmp:x2tmp] = pixelate_face(crop_obj_face)
                crop_obj_face = frame_copy[y1tmp:int(y1tmp+(y2tmp-y1tmp)*0.75),x1tmp:x2tmp]
                frame_copy[y1tmp:int(y1tmp+(y2tmp-y1tmp)*0.75),x1tmp:x2tmp] = pixelate_face(crop_obj_face,max(4,int((y2tmp-y1tmp)/18)))
        
        dist_stat = {}
        for box in boxes:
            id_tensor = box.id
            box_xyxy = np.squeeze(box.xyxy.tolist())
            x_world, y_world = get_world_xy(box_xyxy)
            x1 = int(box_xyxy[0])
            y1 = int(box_xyxy[1])
            x2 = int(box_xyxy[2])
            y2 = int(box_xyxy[3])
            
            if id_tensor is not None:
                id = int(id_tensor.item())
                dist_stat[id] = [x_world, y_world]

                ### SELECT THE INTERSECTING BOXES: HUMAN AND FACE
                if len(mask_results) > 0:
                    mask_status, mask_conf, x1Mask, y1Mask, x2Mask, y2Mask, max_area = -1,0.0,-1,-1,-1,-1,0.0
                    for box_face in mask_results[0].boxes:
                        face_xyxy = np.squeeze(box_face.xyxy.tolist())
                        x1tmp, y1tmp, x2tmp, y2tmp = int(face_xyxy[0]), int(face_xyxy[1]), int(face_xyxy[2]), int(face_xyxy[3])
                        area_tmp = area_intersection(make_square_based_on_width(box_xyxy),face_xyxy)
                        if area_tmp > max_area:
                            x1Mask, y1Mask, x2Mask, y2Mask = x1tmp, y1tmp, x2tmp, y2tmp
                            mask_status = int(box_face.cls.item())
                            mask_conf = box_face.conf.item()
                            max_area = area_tmp
                    if max_area > 0:
                        if id not in mask_dict:
                            mask_dict[id] = [mask_status, mask_conf]
                        ### IF MASK CLASSIFICATION IS THE SAME, UPDATE CONFIDENCE
                        elif mask_dict[id][0] == mask_status:
                            if mask_dict[id][1] < mask_conf:
                                mask_dict[id] = [mask_status,mask_conf]
                        ### ELSE IF -!do- AND NEW DETECTION IS MORE CONFIDENT, UPDATE BOTH
                        elif mask_dict[id][1] < mask_conf:
                            mask_dict[id] = [mask_status, mask_conf]

                ### DECAY MASK_STAT CONFIDENCE
                if id in mask_dict:
                    mask_status = mask_dict[id][0]
                    mask_conf = mask_dict[id][1]
                    mask_dict[id][1] = mask_dict[id][1]-mask_conf_decay

                ### ADD THE CURRENT VALUES TO THE DATAFRAME df
                df.loc[len(df.index)] = [time_frame,id,x1,x2,y1,y2,mask_status,mask_conf,x_world,y_world]

        ### PUTTING TEXT ON THE FRAME
        for i, row in df[df['frame']==time_frame].iterrows():
            x1 = int(row['bbox_xmin'])
            x2 = int(row['bbox_xmax'])
            y1 = int(row['bbox_ymin'])
            y2 = int(row['bbox_ymax'])
            id = int(row['person'])
            closest_dist = 999
            ### COMPUTE THE EUCLIDEAN DISTANCE
            for j in dist_stat:
                d = math.dist(dist_stat[id],dist_stat[j])
                if id!=j and d < closest_dist:
                    closest_dist = d
            if closest_dist >= 1.0:
                if row['mask_stat']==0:
                    text = str(id)+' Unmasked(d='+str(round(closest_dist,1))+')'
                    #bg_color = (0,63,127)
                    bg_color = (0,0,200)
                else: 
                    text = str(id)+' Masked(d='+str(round(closest_dist,1))+')'
                    #bg_color = (0,127,0)
                    bg_color = (200,0,0)
            else:
                if row['mask_stat']==0:
                    text = str(id)+' Unmasked(d='+str(round(closest_dist,1))+')'
                    #bg_color = (0,0,191)
                    bg_color = (0,0,200)
                else: 
                    text = str(id)+' Masked(d='+str(round(closest_dist,1))+')'
                    #bg_color = (191,0,0)
                    bg_color = (200,0,0)
            (w, h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bg_color, thickness=2)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1-2*h), bg_color, -1)
            cv2.putText(frame, text, (x1,y1-10), fontFace, fontScale, color, thickness, lineType)
    
    ### SHOW FRAME
    if showImg_flag:
        cv2.imshow('frame', frame)
        #cv2.imshow('frame', frame_copy)
        cv2.waitKey(1)

    ### SAVE FRAME
    if saveImg_flag:
        vid_out_tracked.write(frame)
        vid_out_blurred.write(frame_copy)

    ### SAVE THE PARTIAL CSV FILE
    if csv_timer<=0:
        if time_frame==0:
            df.to_csv('output/'+fname+'.csv',index=False)
        else:
            df.to_csv('output/'+fname+'.csv', mode='a', index=False, header=False)
        ### RESET THE CSV_TIMER AGAIN (2 mins frame)
        csv_timer = 3600
        ### EMPTY THE DATAFRAME
        df = df[0:0]
    else:
        csv_timer-=1
    time_frame+=1

### SAVE THE REMAINING ROWS FROM THE DATAFRAME
df.to_csv('output/'+fname+'.csv', mode='a', index=False, header=False)
vid_out_tracked.release()
vid_out_blurred.release()
t2 = time.time()
print('Saving Complete. See the output folder. Total execution time:'+str(t2-t1))



