# -*- coding: utf-8 -*-
"""
Tensorflow Implentation of Style-transfer

@author: FAN
"""
import argparse
from style_transfer import Transfer 
import utils
import os

def Argparser():
    
    parser = argparse.ArgumentParser(description='Artisic Style Transfer')
    #train argument
    parser.add_argument('--gpu','-g',default = True,
                        help = 'Use gpu or not.')
    parser.add_argument('--content_img','-C',type = str,
                        help = 'The picture you want to change.')
    parser.add_argument('--content_shape','-C_shape',type = int, default = 512,
                        help = 'The max shape of content img.')
    parser.add_argument('--style_img','-S',type = str,
                        help = 'The style you want to give.')
    parser.add_argument('--style_shape','-S_shape',type = int, default = 512,
                        help = 'The max shape of style img.')
    parser.add_argument('--loss_para', type = float, default = 0.001,
                        help = 'style_loss_para/content_loss_para')
    parser.add_argument('--epoches',type = int ,default = 600,
                        help = 'Train epoches ')
    parser.add_argument('--content_layers' , '-CL', nargs = '*' , type = str ,default = ['conv4_2'],
                        help = 'Layers used to compute content loss.')
    parser.add_argument('--style_layers' , nargs = '*', type = str , default = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'],
                        help = 'Layers used to compute style loss.')
    parser.add_argument('--pool_method' ,'-P', type = str , default = r'avg', choices=['max', 'avg'],
                        help = 'Use max-pooling or avg-pooling.')
    parser.add_argument('--save_path' , type = str , default = r'E:\python.file\git-Repository\TF--style-transfer-collection/dst',
                        help = 'Path to save the output.')

    return parser

'''
style_src_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\style_src\star.jpg'
content_src_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\content_src\girl.jpg'

style_src = get_img(style_src_path,max_size = 512)
content_src = get_img(content_src_path,max_size = 800)

transfer_op = Transfer(content_src,style_src,content_layers = ['conv2_2','conv3_2'],pool_method = 'avg')
img = transfer_op.transfer()
'''

if __name__ == '__main__':
    parser = Argparser()
    args = parser.parse_args()

    content_img = utils.get_img_PIL(args.content_img,max_size = args.content_shape)
    style_img = utils.get_img_PIL(args.style_img,max_size = args.style_shape)
    
    transfer_op = Transfer(content_img = content_img , style_img = style_img ,
                           loss_para = args.loss_para , pool_method = args.pool_method , 
                           content_layers = args.content_layers , style_layers = args.style_layers ,
                           epoches = args.epoches , save_path = args.save_path)
    transfer_op.transfer()



