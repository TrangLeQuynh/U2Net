import argparse
try:
  import onnx
  from onnx_tf.backend import prepare
  import tensorflow as tf
except ImportError as e:
  raise f"Please pip install -r requirements_convert.txt first. {e}"

def onnx2tf(onnx_path, tf_path):
  """
  Convert onnx model to tf.
  onnx_path: ONNX model path to load
  tf_path: TF model path to save
  """
  print("____onnx2tf____")
  onnx_model = onnx.load(onnx_path)

  tf_rep = prepare(onnx_model)
  # export the model to a .pb file
  tf_rep.export_graph(tf_path)

"""
TensorFlow FrozenGraph (.pb) -> TensorFlow Lite(.tflite)
"""
def tf2tflite(tf_path, tflite_path):
  print("____tf2tflite____")
  converter = tf.lite.TFLiteConverter.from_saved_model(tf_path) # from _frozen_graph
  tflite_model = converter.convert()
  with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx", type=str, required=True)
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  
  model_name = args.onnx.split(".")[0]
  onnx2tf(onnx_path=args.onnx, tf_path=f"{model_name}.pb")
  tf2tflite(tf_path=f"{model_name}.pb", tflite_path=f"{model_name}.tflite")
