import time

def readable_time():
    readable_time = time.ctime(time.time())
    readable_time = readable_time.split(" ")
    readable_time = " ".join(readable_time[1:])
    readable_time = readable_time.replace(" ", "_").replace(":", "_")
    return readable_time

