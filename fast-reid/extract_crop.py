import cv2
import os
import random

def process_videos(data_dir, res_dir, interval=500):
    scenes = [scene for scene in os.listdir(data_dir) if scene[-3:] != 'txt' and scene[-4:] != 'json']
    for scene in scenes:
        print(scene)
        os.makedirs(os.path.join(res_dir,scene),exist_ok=True)
        cameras=os.listdir(os.path.join(data_dir,scene))
        cameras.remove('ground_truth.txt')
        with open(os.path.join(data_dir,scene,'ground_truth.txt'),'r') as f:
            annos=f.readlines()
        anno_id=0
        annolen=len(annos)
        cameras.sort()
        for i in range(len(cameras)):
            camera=cameras[i]
            cap=cv2.VideoCapture(os.path.join(data_dir,scene,camera,'video.mp4'))
            totalframe=int(cap.get(7))
            print(camera,totalframe)
            curcamid=int(camera[-4:])
            for frame_id in range(1,totalframe,interval):
                cap.set(1,frame_id)
                ret,frame=cap.read()
                if anno_id>=annolen:
                
                    break
                anno=annos[anno_id]
                anno=[int(i) for i in anno[:-1].split()[:7]]

                while (anno[2]<frame_id+1):
                
                    anno_id+=1
                    if anno_id>=annolen:
                        
                        break
                    anno=annos[anno_id]
                    anno=[int(i) for i in anno[:-1].split()[:7]]
                
                while (anno[2]==frame_id+1):

                    id,x,y,w,h=anno[1],anno[3],anno[4],anno[5],anno[6]
                    anno_id+=1
                    anno=annos[anno_id]
                    if anno_id>=annolen:
                        break
                    anno=[int(i) for i in anno[:-1].split()[:7]]
                
                    if random.random()<0.1:
                        w+=int(random.random()*0.1*w)
                        h+=int(random.random()*0.1*w)
                        x-=int(random.random()*0.1*w)
                        y-=int(random.random()*0.1*w)
                        x=max(0,x)
                        y=max(0,y)
                    crop=frame[y:y+h,x:x+w]
                    cv2.imwrite(os.path.join(res_dir,scene,'{}_{}_{}.jpg'.format(id,i,frame_id)),crop)
            while (anno[2]<=totalframe) and (anno[0]==curcamid):
                anno_id+=1
                if anno_id>=annolen:
                    break
                anno=annos[anno_id]
                anno=[int(i) for i in anno[:-1].split()[:7]]

current_file_path = os.path.abspath(__file__)
path_arr = current_file_path.split('/')[:-2]
root_path = '/'.join(path_arr)

train_data_dir = os.path.join(root_path,'dataset/train')
val_data_dir = os.path.join(root_path,'dataset/val')
res_dir = os.path.join(root_path,'fast-reid/dataset')

# Process train dataset
process_videos(train_data_dir, res_dir)

# Process validation dataset
process_videos(val_data_dir, res_dir)
