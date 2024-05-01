import os

def get_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))

def get_world_size():
    return int(os.getenv("WORLD_SIZE", "1"))