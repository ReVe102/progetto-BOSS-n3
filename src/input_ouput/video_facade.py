class VideoInputFacade:
    def __init__(self, video_source):
        self.video_source = video_source

    def get_video_source(self):
        return self.video_source