"""
This script calculates the mean average precision (mAP) 
for a set of precision-recall values.

It assumes that the bounding boxes are in the format [x1, y1, x2, y2]
and confidence scores are non-existent because the language model 
is not an object detector.

Corrected the case if there are number of detections (predictions) is
greater than or equal the number of ground truths (based on coco annotations)

TODO: Handle "iscrowd=True" cases for the IoU computation

(c) earl-juanico, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
#from scipy.integrate import simps
from scipy.integrate import simpson as simps #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html
import pandas as pd

# Assume that box is in the format [tl_corner(x,y), br_corner(x,y)]
#	Origin is the top-left (tl) corner of the image
def calc_iou(bbox_a, bbox_b):
    x1a,y1a,x2a,y2a = bbox_a
    x1b,y1b,x2b,y2b = bbox_b

    # widths and heights
    w1 = x2a - x1a
    h1 = y2a - y1a
    w2 = x2b - x1b
    h2 = y2b - y1b

    # For a proper bbox, assert the following:
    #assert(w1 >= 0 and h1 >= 0)  # comment out for fuyu cap in 7B llava (TODO: resolve)
    #assert(w2 >= 0 and h2 >= 0)  # comment out for fuyu cap in 7B llava (TODO: resolve)


    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)

    # If the intersection is empty, return 0
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # Calculate the area of intersection and union
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU and return it
    # debug:
    #print(f'bbox_a: {bbox_a}, bbox_b: {bbox_b}, intersection_area: {intersection_area}, union_area: {union_area}, IoU: {intersection_area / union_area}')

    iou = intersection_area / union_area
    return iou

def calculate_ap(points:list[tuple[float,float]])->tuple[float, Image.Image]:
    """
    Calculates the average precision (AP) for a collection of (precision, recall) tuples.
        Input:
            points: a tuple aggregate of (precision, recall) values
        Output:
            AP: the average precision calculated from the PR curve
            image: the PR curve image
    """
    
    x_values = [p[1] for p in points]
    y_values = [p[0] for p in points]
    
    
    # Sort the list of points by ascending order of the second element (recall), 
    #  then by descending order of the first element (precision).
    #sorted_points = sorted(points, key=lambda x: (x[1], -x[0]))

    
    #x_values = [p[1] for p in sorted_points]
    #y_values = [p[0] for p in sorted_points]
    
    #print(f'original: {points}')
    #print(f'x_values: {x_values}')
    #print(f'y_values: {y_values}')
    
    #auc = np.trapz(y_values, x=x_values)  # can be improved
    #auc = simps(y_values, x=x_values)   # Using Simpson's rule
    '''
    # Calculate the area under the curve using precision envelope
    # Create a DataFrame from the x and y values
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    # Group by x values and take the maximum y value for each group
    df = df.groupby('x')['y'].max().reset_index()
    auc = np.trapz(df['y'], x=df['x'])
    #auc = simps(df['y'], x=df['x'])
    x_values = df['x'].tolist()
    y_values = df['y'].tolist()
    '''
    
    #Another technique for obtaining the auc with a precision envelope
    # Source: https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
    #mrec = np.array(x_values)#np.concatenate(([0.], x_values, [1.]))
    #mpre = np.array(y_values)#np.concatenate(([0.], y_values, [0.]))
    #mrec = np.concatenate(([0.], x_values, [1.]))    
    if len(x_values) > 1: # the last value of x (x[-1]) will be partnered to y=0
        mrec = np.concatenate(([0.], x_values, [x_values[-1]]))    
    elif len(x_values) == 1: # the only value of x (x[0]) will be partnered to y=0
        mrec = np.concatenate(([0.], x_values, [x_values[0]]))    
    else:   # there are no values of x
        mrec = np.concatenate(([0.], x_values, [1.]))    
    mpre = np.concatenate(([0.], y_values, [0.]))

    
    for i in range(mpre.size-1,0,-1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        
    #auc = np.trapz(y=mpre.tolist(),x=mrec.tolist())
    k = np.where(mrec[1:] != mrec[:-1])[0]
    auc = np.sum((mrec[k+1] - mrec[k]) * mpre[k+1])
    
    # Plot the PR Curve
    plt.plot(x_values, y_values, '-')
    plt.plot(mrec.tolist(), mpre.tolist(), 'ro--')
    #plt.title(f'Precision-Recall Curve, AP={auc:.2f}')
    plt.xlim([0, 1])  # Set the range of the x axis to [0, 1]
    plt.ylim([0, 1])  # Set the range of the y axis to [0, 1]
    plt.tick_params(axis='both', which='major', labelsize=14)
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #image = Image.open(buf)

    image = plt.gcf()

    return auc, image


def PR(ground_truths:list, 
       predictions:list[tuple[float, float]], 
       iou_threshold=0.5,
       conf_threshold=0.0, # default means all detections considered 
       verbose=True)->tuple[int, int, int, list[float]]:
    IoU = np.zeros(len(predictions))  # IoU of each prediction, initialize as array of zeros
    assert(len(ground_truths)>0)  # There must be at least one ground truth bbox  
    
    if verbose: print(f'IoU size: {len(IoU)}')  # debug
    # The calculation depends on whether there are more predictions than ground truths
    if len(predictions) > len(ground_truths):  # false positives abound
        tp = np.zeros(len(predictions))
        fp = np.ones(len(predictions))
        #fn = np.zeros(len(ground_truths))
        fn = np.ones(len(ground_truths))
        exclude=[] # indices of the predictions that have been matched
        for i, gt in enumerate(ground_truths):
            state=[]
            for j, pred in enumerate(predictions):
                bbox, confidence = pred
                iou = calc_iou(bbox, gt)
                if (iou >= iou_threshold and j not in exclude):
                    #and confidence >= conf_threshold):
                    state.append((j,iou))
                else:
                    state.append((j,0))
            # Find the first index of the prediction with the highest IoU
            #  This is the predicted bbox that is closest to a ground truth bbox
            if verbose:
                print(f'Predictions are more than ground truths in this image:')
                print(f'All grounds: {ground_truths}')  #debug
                print(f'All predicts: {predictions}')  #debug
                print(f'states of all predictions: {state}')  #debug      
            if verbose: print(f'index of best prediction: {k}')  #debug
            # Assign the IoUs for every predicted bbox
            # Expected IoU for each prediction is the highest achieved in the image
            #   due to its proximity to a ground truth bbox
            if verbose: print(f'state: {state}, max_k={k}')  #debug
            if verbose: print(f'state size: {len(state)} vs. prediction size: {len(predictions)}')  # debug
            if sum(x[1] for x in state) > 0:
                k = max(state, key=lambda x: x[1])[0]
                if predictions[k][1] >= conf_threshold: # Check confidence of the best prediction
                    tp[k] = 1
                    fp[k] = 0
                    fn[i] = 0
                    exclude.append(k)
                    IoU[k] = state[k][1]   # IoU of the predicted bbox for a given ground truth
                    if verbose: print(f'IOU of best: {IoU[k]}')  #debug
                #else:
                #    fn[i] = 1
        IoU = IoU.tolist()   # Convert the array to list
        if verbose: print(f'The IoU: {IoU}')  #debug
        TP = int(np.sum(tp))
        FP = int(np.sum(fp))
        FN = int(np.sum(fn))
        #print(f'\n|pd|={len(predictions)}, |gt|={len(ground_truths)} vs. TP={int(TP)}, FP={int(FP)}, FN={int(FN)}')
        #print(f'{exclude}')
        if verbose: print(f'|pd|+|gt| = {len(predictions)+len(ground_truths)} vs. TP+FP+FN={int(TP+FP+FN)}')  #debug
        return TP, FP, FN, IoU
    else: # len(predictions) <= len(ground_truths): # false negatives abound
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        fn = np.ones(len(ground_truths))
        exclude=[]  # indices of the ground truths that have been matched
        state = []
        for i, pred in enumerate(predictions):
            bbox, confidence = pred            
            temp_state=[]
            for j, gt in enumerate(ground_truths):
                iou = calc_iou(bbox, gt)
                if iou >= iou_threshold and i not in exclude: #j not in exclude:
                    temp_state.append((i,iou))
                else:
                    temp_state.append((i,0))
            # Find the first index of the prediction with the highest IoU
            if verbose:
                print(f'Predictions are fewer than or equal to ground truths in this image:') #debug
                print(f'All grounds: {ground_truths}')  #debug
                print(f'All predicts: {predictions}')  #debug
            k = max(temp_state, key=lambda x: x[1])[0]
            if verbose: print(f'index of best prediction: {k}')  #debug
            # Assign the IoUs for every predicted bbox
            # Expected IoU for each prediction is the highest achieved in the image
            #   due to its proximity to a ground truth bbox
            if verbose: print(f'state: {temp_state}')  #debug
            if verbose: print(f'PT:{len(predictions)} vs. GT:{len(ground_truths)}! state size: {len(temp_state)} vs. prediction size: {len(predictions)}')  # debug
            # Correction for |gt| > |pd|
            """
              state_dict is a dictionary where the keys are the first elements of the tuples in state and the values 
               are the maximum of the second elements of the tuples with the same first element.

              list(state_dict.items()) converts this dictionary back to a list of tuples. The result is stored in state
            """
            state_dict={}
            for s in temp_state:
                if s[0] not in state_dict or s[1] > state_dict[s[0]]:
                    state_dict[s[0]]=s[1]
            state.append(list(state_dict.items())[0])              
            if (sum(x[1] for x in temp_state) > 0):# and
                #confidence >= conf_threshold):
                tp[i] = 1
                fn[k] = 0       # something is wrong here
                #exclude.append(k)  # why exclude the ground truth with the best IoU
                #print(f'{k} excluded')
                exclude.append(i)
                if verbose: 
                    print(f'prediction {i} excluded')
                    print(f'IOU of best: {state[k][1]}')  #debug
            else:
                fp[i] = 1
        IoU = [s[1] if s[1] > IoU[m] else 0 for m,s in enumerate(state)]  # Correction for |gt| > |pd| (Array is now a list)
        if verbose: print(f'The IoU: {IoU}')  #debug
        TP = int(np.sum(tp))
        FP = int(np.sum(fp))
        FN = int(np.sum(fn))
        return TP, FP, FN, IoU
