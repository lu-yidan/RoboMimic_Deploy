"""Arrange actual MuJoCo renders into a labelled scene overview (CAD environment)."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,default=ROOT/'outputs/climb_wall_2m_v3')
    args = parser.parse_args()
    for name in ('overview','robot','hands','feet','side'):
        Image.fromarray(np.load(args.scene/(name+'.npy'))).save(args.scene/(name+'.png'))
    canvas = Image.new('RGB',(1700,1380),'#eef1f4')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',35)
    draw = ImageDraw.Draw(canvas)
    full = Image.open(args.scene/'overview.png').crop((370,0,1240,1200))
    full.thumbnail((870,1200));canvas.paste(full,(0,80))
    for name,xy,label in [('hands',(870,80),'双钩对齐初始手杆 · 高 1.55 m'),
                          ('feet',(870,680),'脚底接触横杆 · 高 0.30 m')]:
        im = Image.open(args.scene/(name+'.png'));im.thumbnail((830,550));canvas.paste(im,xy)
        draw.text((xy[0]+15,xy[1]+555),label,font=font,fill='#233448')
    draw.text((25,17),'G1 横杆墙 | 目标净爬升 2.00 m',font=title,fill='#172b40')
    draw.text((25,1282),'14 排横杆 · 间距 25 cm · 直径 30 mm · 墙高 3.8 m · 倾角 5°',font=font,fill='#233448')
    draw.text((25,1327),'已完成场景与静态几何检查；尚未验证机器人连续爬升。',font=font,fill='#526274')
    canvas.save(args.scene/'scene_preview.png')


if __name__ == '__main__':
    main()
