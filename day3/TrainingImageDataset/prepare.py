import os

#####Kelly
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")

#####Kelly
# 替换路径为你本地的 train 和 val 文件夹路径
create_txt_file(r'/Users/weirdprincess/PycharmProjects/dataset/image2/train', 'train.txt')
create_txt_file(r'/Users/weirdprincess/PycharmProjects/dataset/image2/val', 'val.txt')
