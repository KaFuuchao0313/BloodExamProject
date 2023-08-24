#��Ҫ����Ŀ⣬struct��һ���ܳ��õĶ����ƽ�����
import numpy as np 
import struct
 
 def decode_idx3_ubyte(idx3_ubyte_file):#�˺�����������idx3�ļ���idx3_ubyte_filecָ��ͼ���ļ�·��
    #��ȡ����������
    bin_data=open(idx3_ubyte_file,'rb').read()
    #�����ļ�ͷ��Ϣ������Ϊħ����ͼƬ������ÿ��ͼƬ�ߡ�ÿ��ͼƬ��
    offest=0
    fmt_header='>iiii'    magic_number,num_images,num_rows,num_cols=struct.unpack_from(fmt_header,bin_data,offest)
    print('ħ����%d,ͼƬ������%d��ͼƬ��С��%d%d' % (magic_number,num_images,num_rows,num_cols))
    #�������ݼ�
    image_size=num_rows*num_cols
    offest += struct.calcsize(fmt_header)
    fmt_image='>'+str(image_size)+'B'
    images=np.empty((num_images,num_rows,num_cols))
    for i in range(num_images):
        if (i+1)%10000==0:
            print('�ѽ���%d'%(i+1)+'��')        images[i]=np.array(struct.unpack_from(fmt_image,bin_data,offest)).reshape((num_rows,num_cols))
        offest+=struct.calcsize(fmt_image)
    return images
'''images��һ����ά����,images[i][a][b]��ʾ��i��ͼƬ�ĵ�����a�У�b�е�����'''
def decode_idx1_ubyte(idx1_ubyte_file):#����idx1�ļ�������idx1_ubyte_fileָ����ǩ�ļ�·��
    #��ȡ����������
    bin_data=open(idx1_ubyte_file,'rb').read()
    #�����ļ�ͷ��Ϣ������Ϊħ���ͱ�ǩ��
    offest=0
    fmt_header='>ii'
    magic_number,num_images=struct.unpack_from(fmt_header,bin_data,offest)
    print('ħ����%d��ͼƬ������%d��' % (magic_number,num_images))
    #�������ݼ�
    offest+=struct.calcsize(fmt_header)
    fmt_image='>B'
    labels=np.empty(num_images)
    for i in range(num_images):
        if (i+1)%10000==0:
            print('�ѽ�����%d'%(i+1)+'��')
        labels[i]=struct.unpack_from(fmt_image,bin_data,offest)[0]
        offest+=struct.calcsize(fmt_image)
    print(labels[0])
    return labels
'''labels��һ��һά���飬ÿ��Ԫ�ض�һһ��Ӧimages[i]'''
