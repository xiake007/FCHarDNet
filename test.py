import torch
import argparse
import os
import numpy as np

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2


def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader("cityscapes")
    loader = data_loader(
        root='./Cityscapes',
        is_transform=True,
        img_size=eval(args.size),
        test_mode=True
    )
    n_classes = loader.n_classes

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(args.model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model, loader

def test(args):
    device, model, loader = init_model(args)
    bisenet_model=load_bise_model()
    proc_size = eval(args.size)

    if os.path.isfile(args.input):
        img_raw, decoded = process_img(args.input, proc_size, device, model, loader)
        blend = np.concatenate((img_raw, decoded), axis=1)
        out_path = os.path.join(args.output, os.path.basename(args.input))
        cv2.imwrite("test.png", decoded)
        cv2.imwrite(out_path, blend)

    elif os.path.isdir(args.input):
        print("Process all image inside : {}".format(args.input))

        for img_file in os.listdir(args.input):
            _, ext = os.path.splitext(os.path.basename((img_file)))
            if ext not in [".png", ".jpg"]:
                continue
            img_path = os.path.join(args.input, img_file)

            img, decoded = process_img(img_path, proc_size, device, model, loader)
            bi_img, bi_decoded = bi_process_img(img_path, proc_size, device, bisenet_model, loader)
            blend = np.concatenate((img, decoded), axis=1)
            bi_blend = np.concatenate((bi_img, bi_decoded), axis=1)
            full_blend = np.concatenate((blend, bi_blend), axis=0)
            out_path = os.path.join(args.output, os.path.basename(img_file))
            cv2.imwrite(out_path, blend)
            # cv2.imshow('test',full_blend.astype(np.uint8))
            # if cv2.waitKey(0)==27:
            #     continue

def bi_process_img(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)[0]
    # import time
    # for i in range(100):
    #     outputs = model(images)[0]
    #     if i==10:
    #         start=time.time()
    # print(f'bise-net cost time: {(time.time()-start)/90}')
    pred = outputs.argmax(dim=1).squeeze().detach().cpu().numpy()
    decoded = loader.decode_segmap(pred)
    decoded = decoded*std+mean

    return img_resized, decoded

def process_img(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    # import time
    # for i in range(100):
    #     outputs = model(images)
    #     if i==10:
    #         start=time.time()
    # print(f'hard-net cost time: {(time.time()-start)/90}')
    parsing = np.squeeze(outputs.data.max(1)[1].cpu().numpy(),axis=0)
    decoded = loader.decode_segmap(parsing)
    decoded = decoded*std+mean

    return img_resized, decoded


def main():
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        # required=True,
        default="./models/hardnet70_cityscapes_model.pkl",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="icboard",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--size",
        type=str,
        default="540,960",
        help="Inference size",
    )

    parser.add_argument( "--input", nargs="?", type=str, default='./imgs', help="Path of the input image/ directory" )
    # parser.add_argument( "--input", nargs="?", type=str, default=r"F:\archive\dataset\dvr\dvr_out_imgs_5k\data", help="Path of the input image/ directory" )
    parser.add_argument(
        "--output", nargs="?", type=str, default="./output", help="Path of the output directory"
    )
    args = parser.parse_args()
    test(args)

import os.path as osp
def rewrite_imgs():
    imgs_root=r"F:\archive\dataset\dvr\dvr_out_imgs_5k"
    dst_dir=osp.join(imgs_root,'org')
    src_dir=osp.join(imgs_root,'data')
    if not osp.exists(src_dir):
        os.mkdir(src_dir)
    for img_name in os.listdir(dst_dir):
        img = osp.join(dst_dir,img_name)
        img_data = cv2.imread(img)
        if img_data is None:
            print(f'img file is not exist: {img_name}')
            continue
        new_name=img_name+'.jpg'
        cv2.imwrite(osp.join(src_dir,new_name),img_data)

def show_img():
    data_root='./output'
    for img in os.listdir(data_root):
        img=cv2.imread(osp.join(data_root,img))
        cv2.imshow('test',img)
        if cv2.waitKey(0)==27:
            continue

def load_bise_model():
    import bisenetv2
    model=bisenetv2.BiSeNetV2(n_classes=19).to(torch.device("cpu"))
    state_dict=torch.load('models/bisenet_vmodel_final.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model
if __name__ == "__main__":
    main()
    # show_img()
    # load_model()

