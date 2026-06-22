from utils import read_video, save_video
from trackers import Tracker
def main():
    frames = read_video("data/input/08fd33_4.mp4")
    
    tracker = Tracker("models/best.pt")
    
    tracks = tracker.get_obj_tracks(frames,
                           read_from_stub=True,
                           read_path="data/output/08fd33_4_tracks.pkl")
    
    output_video_frames = tracker.draw_annotations(frames, tracks)
    
    save_video(output_video_frames, "data/output/08fd33_4_output.mp4")
    
if __name__ == "__main__":
    main()