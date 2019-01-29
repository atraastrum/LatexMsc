#define ONNX_ML 1
#include <onnx/onnx_pb.h>
#include <onnx/onnx-ml.pb.h>
#include <iostream>
#include <fstream>

int main() {
  onnx::ModelProto model;
  std::ifstream model_file{"simple_onnx_export.onnx", std::ios_base::binary};
  model.ParseFromIstream(&model_file);

  std::cout << model.DebugString() << std::endl;
}
