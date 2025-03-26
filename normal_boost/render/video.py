"""
    Helper class to create and add images to video
"""
import imageio
import numpy as np
import os

class Video():
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:
        
        if path[-1] != "/":
            path += "/"
        
        full_path = path + name
        ext = os.path.splitext(full_path)[-1].lower()

        # Accepted video formats
        video_exts = ['.mp4', '.avi', '.mov', '.gif']
        
        if ext not in video_exts:
            full_path += ".mp4"
            
        self.writer = imageio.get_writer(
            full_path,
            mode=mode,
            fps=fps,
            codec=codec,
            bitrate=bitrate
        )
    
    def ready_image(self, image, write_video=True):
        # assuming channels last - as renderer returns it
        if len(image.shape) == 4: 
            image = image.squeeze(0)[..., :3].detach().cpu().numpy()
        else:
            image = image[..., :3].detach().cpu().numpy()

        image = np.clip(np.rint(image*255.0), 0, 255).astype(np.uint8)

        if write_video:
            self.writer.append_data(image)

        return image

    def close(self):
        self.writer.close()