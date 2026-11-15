import numpy as np

pred = np.random.rand(3339)
target = np.random.rand(3339)

def w(depth, mu=35, sigma=50):
    return np.exp(-((depth - mu) ** 2) / (2 * sigma ** 2))


def weight_rmse(llh,pred,target):
    # llh = np.loadtxt('./wj0925/data/fault/llh')
    depth = -llh[:,-1]
    weight = w(depth)
    weight = torch.as_tensor(weight).to(device, dtype=torch.float)
    # print(weight.shape)
    diff_norms = torch.sqrt(torch.sum(weight*(pred - target)**2))  # 对后两维计算L2范数
    n_norms = torch.sqrt(torch.as_tensor(pred.shape[0]).to(device, dtype=torch.float))  # 对目标值的L2范数
    weight_rmse_result = diff_norms / n_norms  # 计算相对误差的均值
    # weight_rmse_result = diff_norms
    return weight_rmse_result

def weight_acc(llh,pred,target):
    depth = -llh[:,-1]
    weight = w(depth)
    weight = torch.as_tensor(weight).to(device, dtype=torch.float)
    diff_norms = torch.abs(torch.sum(weight*(pred*target))) # 对后两维计算L2范数
    x_norms = torch.sum(weight*(pred*pred))
    y_norms = torch.sum(weight*(target*target))
    weight_acc_result = diff_norms / torch.sqrt((x_norms*y_norms))  # 计算相对误差的均值
    return weight_acc_result

def quantiles(x):
    ''' helper to compute quantiles based on qtile of the field '''
    qs = 100
    qlim = 4
    qcut = 1
    qtile = 1. - np.logspace(-qlim, -qcut, num=qs)
    return np.quantile(x, q=qtile, axis=-1).squeeze()

def RQE(pred_q,targ_q):
    return np.sum((pred_q - targ_q)/targ_q)
