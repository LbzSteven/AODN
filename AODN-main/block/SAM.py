import numpy as np
import numpy

def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    h,w,c=x_pred.shape
    sam_rad = []
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()

            s = np.sum(np.dot(tmp_pred, tmp_true))
            t = (np.sqrt(np.sum(tmp_pred ** 2))) * (np.sqrt(np.sum(tmp_true ** 2)))
            th = np.arccos(s/t)


            sam_rad.append(th)
    sam_deg = np.mean(sam_rad)
    return sam_deg

# def SAM(x,y):
#
#     # print(s,t)
#     return th


if __name__ == '__main__':
    # dn_block = dn_block(alpha=0.1)
    # summary(dn_block.cuda(), input_size=[(60, 20, 20)], batch_size=1, device="cuda")
    a=numpy.zeros([10,20,30])
    print(a)
    b=numpy.zeros([10,20,30])
    print(sam(a,b))