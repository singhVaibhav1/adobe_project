import numpy as np
import cv2
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

def load_points_from_csv(file_path):
    try:
        df=pd.read_csv(file_path,header=None)
        points=df.iloc[:,2:4].values
        return points
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def normalize_points(points,image_size=(500,500)):
    x_min,x_max=np.min(points[:,0]),np.max(points[:,0])
    y_min,y_max=np.min(points[:,1]),np.max(points[:,1])
    points[:,0]=(points[:,0]-x_min)/(x_max-x_min)*(image_size[1]-1)
    points[:,1]=(points[:,1]-y_min)/(y_max-y_min)*(image_size[0]-1)
    return points

def create_image_from_points(points,image_size=(500,500)):
    image=np.zeros(image_size,dtype=np.uint8)
    for x,y in points:
        if 0<=int(x)<image_size[1] and 0<=int (y)<image_size[0]:
            image[int (y),int(x)]=255
    return image
    
def detect_shapes(image):
    blurred=cv2.GaussianBlur(image,(5,5),0)
    edges=cv2.Canny(blurred,50,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    shapes=[]
    for contour in contours:
        epsilon=0.02*cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,epsilon,True)
        x,y,w,h=cv2.boundingRect(contour)
        if len(approx)==3:
            shapes.append(("Triangle",contour))
        elif len(approx)==4:
            aspect_ratio=float(w)/h
            if 0.95<=aspect_ratio<=1.05:
                shapes.append(("Square",contour))
            else:
                shapes.append(("Rectangle",contour))
        elif len(approx)>4:
            circularity=4*np.pi*cv2.contourArea(contour)/(cv2.arcLength(contour,True)**2)
            if circularity>0.9:
                shapes.append(("Circle",contour))
            else:
                shapes.append(("Ellipse",contour))
        else:
            shapes.append(("Polygon",contour))
    return shapes

def detect_stars(image):
    blurred=cv2.GaussianBlur(image,(5,5),0)
    edges=cv2.Canny(blurred,50,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    stars=[]
    for contour in contours:
        epsilon=0.02*cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,epsilon,True)
        num_sides=len(approx)
        if num_sides>=5:
            M=cv2.moments(contour)
            if M["m00"]!=0:
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])

                distances=[np.sqrt((pt[0][0]-cx)*2+(pt[0][1]-cy)*2)for pt in approx]
                avg_distance=np.mean(distances)
                distance_variance=np.var(distances)
                if distance_variance<0.1*avg_distance:
                    stars.append(("Star",contour))
    return stars

def draw_symmetry_lines(image,shapes):
    for shape,contour in shapes:
        if shape in ["Circle","Ellipse"]:
            (x,y),(MA,ma),angle=cv2.fitEllipse(contour)
            major_axis=int(MA/2)
            minor_axis=int (ma/2)
            center=(int (x),int(y))
            cv2.line(image,(center[0]-major_axis,center[1]),(center[0]+major_axis,center[1]),(255,0,0),2)
            cv2.line(image,(center[0],center[1]-minor_axis),(center[0],center[1]+minor_axis),(255,0,0),2)
        elif shape in["Square","Rectangle"]:
            x,y,w,h=cv2.boundingRect(contour)
            cv2.line(image,(x+w//2,y),(x+w//2,y+h),(255,0,0),2)
            cv2.line(image,(x,y+h//2),(x+w,y+h//2),(255,0,0),2)

        elif shape in["Polygon","Star"]:
            M=cv2.moments(contour)
            if M["m00"]!=0:
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])
                for pt in contour:
                    cv2.line(image,(cx,cy),tuple(pt[0]),(255,0,0),2)
    return image

def visualize_shapes(image,shapes):
    color_image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for shape,contour in shapes:
        cv2.drawContours(color_image,[contour],-1,(0,255,0),2)

    color_image=draw_symmetry_lines(color_image,shapes)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB))
    plt.title('Detected shapes with symmetry Lines')
    plt.axis('off')
    plt.show()

def complete_curves(image):
    blurred=cv2.GaussianBlur(image,(5,5),0)
    edges=cv2.Canny(blurred,50,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    completed_curves=[]

    for contour in contours:
        epsilon=0.02*cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,epsilon,True)
        if len(approx)>=5:
            points=np.array(approx).reshape(-1,2)
            t=np.linspace(0,1,len(points))
            tck,u=interpolate.splprep([points[:,0],points[:, 1]],u=t,s=0,per=True)
            unew=np.linspace(0,1,100)
            out=interpolate.splev(unew,tck)
            curve=np.vstack(out).T.astype(np.int32)
            completed_curves.append(("Completed Curve",curve))
        else:
            print("This shape is asymmetric and cannot be regularized.")
    return completed_curves

def process_image(image):
    shapes=detect_shapes(image)
    visualize_shapes(image,shapes)
    completed_curves=complete_curves(image)

    for curve_type,curve_points in completed_curves:
        plt.plot(curve_points[:,0],curve_points[:,1],label=curve_type)
    plt.legend()
    plt.show()

file_path='occlusion2.csv'
points=load_points_from_csv(file_path)
if points is not None:
    normalized_points=normalize_points(points)
    image=create_image_from_points(normalized_points)
    process_image(image)
    
                    
