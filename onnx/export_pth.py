import argparse
from Net.colorNet import myNet_ocr_color
from alphabets import plate_chr
import torch
import onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/plate_rec_ocr.pth', help='weights path')
    parser.add_argument('--save_path', type=str, default='../weights/plate_rec_ocr.onnx', help='onnx save path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[48, 168], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--simplify', action='store_true', default=False, help='simplified onnx')

    opt = parser.parse_args()
    print(opt)
    checkpoint = torch.load(opt.weights)
    cfg = checkpoint['cfg']
    model = myNet_ocr_color(num_classes=len(plate_chr), cfg=cfg, export=True, color_num=5)
    model.load_state_dict(checkpoint['state_dict'],False)
    model.eval()

    input = torch.randn(opt.batch_size, 3, 48, 168)
    onnx_file_name = opt.save_path

    torch.onnx.export(model, input, onnx_file_name,
                      input_names=["images"], output_names=["output_1", "output_2"],
                      verbose=False,
                      opset_version=11,
                      dynamic_axes={'images': {0: 'batch'},
                                    'output': {0: 'batch'}
                                    } if opt.dynamic else None)
    print(f"convert completed,save to {opt.save_path}")
    if opt.simplify:
        from onnxsim import simplify

        print(f"begin simplify ....")
        input_shapes = {"images": list(input.shape)}
        onnx_model = onnx.load(onnx_file_name)
        model_simp, check = simplify(onnx_model, test_input_shapes=input_shapes)
        onnx.save(model_simp, onnx_file_name)
        print(f"simplify completed,save to {opt.save_path}")
