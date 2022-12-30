import torch

def IoU(a:torch.tensor,b:torch.tensor):
    for vec in a:
        box1_x1, box1_y1 = vec[0], vec[1]
        box1_x2, box1_y2 = vec[2], vec[3]
    for vec in b:
        box2_x1, box2_y1 = vec[0], vec[1]
        box2_x2, box2_y2 = vec[2], vec[3]

    x1, y1 = torch.max(box1_x1,box2_x1), torch.max(box1_y1,box2_y1)
    x2, y2 = torch.min(box1_x2,box2_x2), torch.min(box1_y2,box2_y2)

    INTERSECTION = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a_area = abs((box1_x2 - box1_x1)*(box1_y2 - box1_y1))
    b_area = abs((box2_x2 - box2_x1)*(box2_y2 - box2_y1))
    return INTERSECTION / (a_area + b_area - INTERSECTION + 1e-8)

def Dice(a:torch.tensor,b:torch.tensor):
    TP = len(torch.nonzero(torch.logical_and(a,b)))
    SUM = len(torch.nonzero(a)) + len(torch.nonzero(b))
    return 2 * TP / SUM

if __name__ == '__main__':
    a = torch.tensor([1,0,1,0])
    b = torch.tensor([1,0,0,0])
    print(Dice(a,b))