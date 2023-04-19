import argparse
import torch
import os
from model.u2net import U2NET, U2NETP
from tools.utils import check_size
try:
  import onnx
  import onnxruntime
except ImportError as e:
  raise ImportError(f'Please install onnx and onnxruntime first. {e}')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="u2netp", required=True)
  parser.add_argument("--gpu", action="store_true", default="False")
  # parser.add_argument("--jit", action="store_true", default="False")
  return parser.parse_args()

def verify_onnx_model(model_path):
  onnx_model = onnx.load(model_path)
  #check that the model is well formed
  onnx.checker.check_model(onnx_model)

  #representation of the graph
  # print(onnx.helper.printable_graph(onnx_model.graph))

def convert_to_onnx(model, save_path):
  input = torch.randn(1, 3, 320, 320)
  with torch.no_grad():
    torch.onnx.export(
      model,
      input,
      save_path,
      export_params=True,
      opset_version=11,#default 13
      do_constant_folding=True, #to execute constant folding for optimization
      verbose=False,
      input_names=["img"],
      output_names=["d0", "d1", "d2", "d3", "d4", "d5", "d6"],
      # output_names=["d0"],
      # dynamic_axes={}
    )

def build_model(args):
    # model_name=  'u2net'#u2netp
    model_name = args.model
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
        
    if args.gpu is True:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # if args.jit is True:
    #     net_jit_path = os.path.join('saved_models', model_name, model_name + '_jit.pt')
    #     if os.path.isfile(net_jit_path) is False:
    #         input = torch.rand(1, 3, 320, 320)
    #         net = torch.jit.trace(net, input.cuda() if args.gpu is True else input, strict=False)
    #         torch.jit.save(net, net_jit_path)
    #     else:
    #         del net
    #         net = torch.jit.load(net_jit_path)
    check_size(model=net)
    return net

if __name__ == "__main__":
  args = parse_args()
  net = build_model(args=args)

  onnx_path = f"{args.model}.onnx"
  convert_to_onnx(net, onnx_path)
  verify_onnx_model(onnx_path)