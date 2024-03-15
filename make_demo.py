import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)
parser.add_argument('--delete_img', action='store_true')

args = parser.parse_args()
dump_path = f'./nav_res/dump/{args.exp_name}/episodes/'
threads_path = sorted(os.listdir(dump_path))

for thread in threads_path:
    i = int(thread.split('_')[1])
    episodes_path = sorted(os.listdir(os.path.join(dump_path, thread)))

    for episode in episodes_path:
        j = int(episode.split('_')[1])
        img_path = os.path.join(dump_path, thread, episode)

        os.system(f"ffmpeg -r 3 -i {img_path}/{i}-{j}-Vis-%d.png {img_path}/demo.mp4")
        if args.delete_img:
            imgs = sorted(os.listdir(img_path))
            for img in imgs:
                if img.endswith('.png'):
                    os.remove(os.path.join(img_path, img))