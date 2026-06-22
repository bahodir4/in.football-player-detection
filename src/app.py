from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
import numpy as np
import cv2

def main():
    video_frames = read_video('data/input/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path="stubs/camera_movement.pkl")
    
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)
     
    view_transform = ViewTransformer()
    view_transform.add_transformed_position_to_tracks(tracks) 
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control= np.array(team_ball_control)
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    save_video(output_video_frames, "data/output/08fd33_4_output.mp4")
    
if __name__ == "__main__":
    main()