import numpy as np, scipy.io as sio, os
import onnx
from onnx import helper, TensorProto, numpy_helper

os.makedirs('models', exist_ok=True)

def make_net(cin, cout, fname):
    # input [N,cin,D,H,W] -> conv 1x1x1 -> [N,cout,D,H,W], spatial preserved
    W = np.random.randn(cout, cin, 1, 1, 1).astype(np.float32) * 0.1
    B = np.zeros((cout,), np.float32)
    Wi = numpy_helper.from_array(W, 'W')
    Bi = numpy_helper.from_array(B, 'B')
    node = helper.make_node('Conv', ['input','W','B'], ['output'],
                            kernel_shape=[1,1,1], pads=[0,0,0,0,0,0])
    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                        ['n', cin, 'd','h','w'])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                        ['n', cout, 'd','h','w'])
    graph = helper.make_graph([node], 'g', [inp], [out], [Wi, Bi])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, f'models/{fname}')
    print('wrote', fname)

make_net(1,1,'240904_QSMnet.onnx')
make_net(1,1,'R2PRIMEnet.onnx')
make_net(3,2,'chi_sepnet.onnx')

nf = dict(field_mean=0.0, field_std=0.05, r2prime_mean=0.05, r2prime_std=0.05,
    r2star_mean=0.2, r2star_std=0.1, x_pos_mean=0.0, x_pos_std=0.05,
    x_neg_mean=0.0, x_neg_std=0.05, cosmos_sus_mean=0.0, cosmos_sus_std=0.05)
sio.savemat('models/norm_factor.mat',
            {k: np.array([[v]], np.float64) for k,v in nf.items()})
print('wrote norm_factor.mat')
