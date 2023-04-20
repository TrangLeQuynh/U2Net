import argparse
import os
try:
  import onnx
  from onnx_tf.backend import prepare
  import tensorflow as tf
except ImportError as e:
  raise f"Please pip install -r requirements_convert.txt first. {e}"
#metadata
import flatbuffers
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata

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

"""Creates the metadata for an U2NetP."""
#Example: https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py
def create_metadata_tflite():
  """ Creates model info."""
  model_meta = _metadata_fb.ModelMetadataT()
  model_meta.name = "u2netp"
  model_meta.description = ("U2Net is a machine learning model that allows you to crop objects in a single shot." 
                            "Take an image as input, it can compute an alpha value to separate the background from the panoramic view.")
  # model_meta.version = ""
  # model_meta.author = ""
  # model_meta.license = ("")

  """ Creates input info. """
  input_meta = _metadata_fb.TensorMetadataT()
  input_meta.name = "image"

  """ Creates output info. """
  output_meta = _metadata_fb.TensorMetadataT()
  output_meta.name = "pred"

  """ Creates subgraph info. """
  subgraph = _metadata_fb.SubGraphMetadataT()
  subgraph.inputTensorMetadata = [input_meta]
  subgraph.outputTensorMetadata = [output_meta]
  model_meta.subgraphMetadata = [subgraph]

  b = flatbuffers.Builder(0)
  b.Finish(
      model_meta.Pack(b),
      _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  metadata_buf = b.Output()
  return metadata_buf

def add_metadata(tflite_path):
  metadata_buf = create_metadata_tflite()

  """Populates metadata and label file to the model file."""
  populator = _metadata.MetadataPopulator.with_model_file(tflite_path)
  populator.load_metadata_buffer(metadata_buf)
  populator.populate()

  verify_metadata(tflite_path=tflite_path)

def verify_metadata(tflite_path):
  # Validate the output model file by reading the metadata and produce
  # a json file with the metadata under the export path
  displayer = _metadata.MetadataDisplayer.with_model_file(tflite_path)
  json_file = displayer.get_metadata_json()

  ### save metadata into json
  # export_json_file = model_name + ".json"
  # with open(export_json_file, "w") as f:
  #   f.write(json_file)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx", type=str, required=True)
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  
  model_name = args.onnx.split(".")[0]
  onnx2tf(onnx_path=args.onnx, tf_path=f"{model_name}.pb")

  tflite_path = f"saved_models/{model_name}.tflite"
  tf2tflite(tf_path=f"{model_name}.pb", tflite_path=tflite_path)
  add_metadata(tflite_path)
  