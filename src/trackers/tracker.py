from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import sys
import os
sys.path.append("../")
from utils import get_center_of_bbox, get_width_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections.extend(detections_batch)
        return detections

    def get_obj_tracks(self, frames, read_from_stub=False, read_path=None):
        
        if read_from_stub and read_path is not None and os.path.exists(read_path):
            with open(read_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)
        
        tracks = {
            "player": [],
            "referee": [],
            "ball": [], 
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for obj_index, cls_id in enumerate(detection_supervision.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    detection_supervision.class_id[obj_index] = cls_names_inv["player"]
                    
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks['player'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})
            
            for i in range(len(detection_with_tracks)):
                bbox = detection_with_tracks.xyxy[i].tolist()
                cls_id = detection_with_tracks.class_id[i]
                track_id = detection_with_tracks.tracker_id[i]

                if cls_names[cls_id] == "player":
                    tracks['player'][frame_num][track_id] = {'bbox': bbox}

                if cls_names[cls_id] == "referee":
                    tracks['referee'][frame_num][track_id] = {'bbox': bbox}

            for i in range(len(detection_supervision)):
                bbox = detection_supervision.xyxy[i].tolist()
                cls_id = detection_supervision.class_id[i]

                if cls_names[cls_id] == "ball":
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if read_path is not None:
            with open(read_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        return frame
        
    def draw_annotations(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
             
            player_dict = tracks['player'][frame_num]
            referee_dict = tracks['referee'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)
                
            output_frames.append(frame)
        
        return output_frames