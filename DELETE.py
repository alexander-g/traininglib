import warnings; warnings.simplefilter('ignore')
import torch
from traininglib import onnxlib


torch.manual_seed(65)

m = torch.nn.Sequential(
    torch.nn.Conv2d(3,5, kernel_size=3),
    torch.nn.BatchNorm2d(5),
    torch.nn.Conv2d(5,1, kernel_size=1),
).train()
print(m[1].running_mean)

x = torch.randn([2,3,10,10])
loss_func = lambda y,t: ( (y-1)**2 ).mean()

fx, sd_grad_vals, sd_nongrad_vals, x \
    = onnxlib.export_model_as_functional_training_onnx(m, loss_func, x)

print(m[1].running_mean)
print(sd_nongrad_vals)

bn_out = None
for node in list(fx.graph.nodes):
    if node.target == torch.ops.aten.native_batch_norm_backward.default:
        bn_out = node
        print(node.__dict__)
        with fx.graph.inserting_after(node):
            new_node = fx.graph.output([bn_out]+[node.args]+[None]*17)
        break
    

fx.graph.lint()
fx.recompile()


print(fx)

out = fx(sd_grad_vals, sd_nongrad_vals, x)


dx,dw,db = out[0][0]
print(dx.shape, dw.shape, db.shape)
ins = out[0][1]
print([x.shape for x in ins[:7]])
(
    grad_out_, input, weight, running_mean, running_var, 
    save_mean, save_invstd, train, eps, grad_input_mask
) = ins


print()

xmu  = (input - save_mean[None,:,None,None])  #cached?
xhat = xmu  * save_invstd[None,:,None,None]   #cached?
dL_dweight = torch.sum(grad_out_ * xhat, dim=[0,2,3])
print(dL_dweight)
print(dw)
assert torch.allclose(dL_dweight, dw)

dL_dbias = torch.sum(grad_out_, dim=[0,2,3])
assert torch.allclose(dL_dbias, db)
print()

print(grad_out_.shape, weight.shape)

N, C, H, W = input.shape
num_elements = N * H * W
dx_hat = grad_out_ * weight[None,:,None,None]
dxmu1  = dx_hat * save_invstd[None,:,None,None]
divar  = torch.sum(dx_hat*xmu, dim=[0,2,3])
#6
sqrtvar = 1. / save_invstd
dsqrtvar = -1. /(sqrtvar**2) * divar
#5
#dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
dvar = 0.5 * save_invstd * dsqrtvar
#step4
#dsq = 1. /N * np.ones((N,D)) * dvar
dsq = 1. /num_elements * torch.ones((N,C,H,W)) * dvar[None,:,None,None]
#step3
dxmu2 = 2 * xmu * dsq
#step2
dx1 = (dxmu1 + dxmu2)
#dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
dmu = -1 * torch.sum(dxmu1+dxmu2, dim=[0,2,3])
#step1
#dx2 = 1. /N * np.ones((N,D)) * dmu
dx2 = 1. /num_elements * torch.ones((N,C,H,W)) * dmu[None,:,None,None]
#step0
dL_dx = dx1 + dx2


print()
print(dL_dx.shape)
print(dx.sum(dim=[2,3]))
print(dL_dx.sum(dim=[2,3]))

assert torch.allclose(dx, dL_dx)


print()

onnxlib.replace_all_aten_native_batch_norm_backward(fx)
out = fx(sd_grad_vals, sd_nongrad_vals, x)
dx2,dw2,db2 = out[0][0]

print(db)
print(db2)
print(dw)
print(dw2)
print()
print(dx.sum(dim=[2,3]))
print(dx2.sum(dim=[2,3]))

print('-'*40)

onnxlib.replace_all_aten_native_batch_norm(fx)
print(fx)
out = fx(sd_grad_vals, sd_nongrad_vals, x)
dx3,dw3,db3 = out[0][0]

print(db3)
print(dw3)
print(dx3.sum(dim=[2,3]))

print('ok')

