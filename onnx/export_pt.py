import argparse
import sys
import time
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import onnx

sys.path.append('../')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/plate_rec.pt', help='weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
    # =======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=False, help='fp16 infer')
    # ================================================================
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))
    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid = [torch.zeros(1)] * 3
    model.model[-1].export_cat = True
    model.eval()
    labels = model.names

    # Checks
    gs = int(max(model.stride))
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        if isinstance(m, models.common.ShuffleV2Block):
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
        if isinstance(m, models.common.BlazeBlock):
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
        if isinstance(m, models.common.DoubleBlazeBlock):
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
    y = model(img)

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, img, f, verbose=False, opset_version=12,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={'input': {0: 'batch'},
                                    'output': {0: 'batch'}
                                    } if opt.dynamic else None)

    # Checks
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    print('ONNX export success, saved as %s' % f)
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

    # onnx infer
    if opt.onnx_infer:
        import onnxruntime
        import numpy as np

        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(f, providers=providers)
        im = img.cpu().numpy().astype(np.float32)  # torch to numpy
        y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
        print("pred's shape is ", y_onnx.shape)
        print("max(|torch_pred - onnx_pred|） =", abs(y.cpu().numpy() - y_onnx).max())

    # TensorRT export
    if opt.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT

        print('\nStarting TensorRT...')
        ONNX_to_TRT(onnx_model_path=f, trt_engine_path=f.replace('.onnx', '.trt'), fp16_mode=opt.fp16_trt)

    # PB export
    if opt.onnx2pb:
        print('download the newest onnx_tf by https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf')
        from onnx_tf.backend import prepare
        import tensorflow as tf

        outpb = f.replace('.onnx', '.pb')
        tf_rep = prepare(onnx_model, strict=False)
        tf_rep.export_graph(outpb)

        out_onnx = tf_rep.run(img)

        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(outpb, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                input_x = sess.graph.get_tensor_by_name(input_names[0] + ':0')
                outputs = []
                for i in output_names:
                    outputs.append(sess.graph.get_tensor_by_name(i + ':0'))
                out_pb = sess.run(outputs, feed_dict={input_x: img})

        print(f'out_pytorch {y}')
        print(f'out_onnx {out_onnx}')
        print(f'out_pb {out_pb}')
