//
// Created by kaylor on 6/20/24.
//

#include "rk3588.h"

#include <fstream>

#include "kaylordut/log/logger.h"

uint16_t Rk3588::instance_count_ = 0;
Rk3588::Rk3588(bool zero_copy) {
  core_mask_ = static_cast<rknn_core_mask>(instance_count_ % 3);
  instance_count_++;
  zero_copy_ = zero_copy;
}

Rk3588::Rk3588(rknn_context *ctx_in, bool zero_copy) {
  dup_ctx_ = ctx_in;
  core_mask_ = static_cast<rknn_core_mask>(instance_count_ % 3);
  instance_count_++;
  zero_copy_ = zero_copy;
}

Rk3588::~Rk3588() {
  if (input_ != nullptr) {
    delete[] input_;
  }
  if (output_ != nullptr) {
    delete[] output_;
  }
  if (input_attr_ != nullptr) {
    delete[] input_attr_;
  }
  if (output_attr_ != nullptr) {
    delete[] output_attr_;
  }
  rknn_destroy(ctx_);
}

void Rk3588::Initialize(const char *model_path) {
  KAYLORDUT_LOG_INFO("model path is {}", std::string(model_path));
  config_.model_name_path = model_path;
  config_.model_format = ModelFormat::RKNN_FORMAT;
  config_.rknn_zero_copy = zero_copy_;
  std::ifstream file(model_path, std::ios::binary);
  assert(file.good());
  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *model_stream = new char[size];
  assert(model_stream);
  file.read(model_stream, size);
  file.close();
  int ret = 0;
  if (dup_ctx_ == nullptr) {
    ret = rknn_init(&ctx_, model_stream, size, 0, NULL);
    delete[] model_stream;
  } else {
    ret = rknn_dup_context(dup_ctx_, &ctx_);
  }
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_init/rknn_dup_context failed, ret={}", ret);
    exit(EXIT_FAILURE);
  }
  config_.rknn_ctx = ctx_;
  switch (core_mask_) {
    case 0:
      ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);
      break;
    case 1:
      ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_1);
      break;
    case 2:
      ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_2);
      break;
    default:
      ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);
  }
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_set_core_mask failed! error code = {}", ret);
    exit(EXIT_FAILURE);
  }
  rknn_sdk_version version;
  ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
  KAYLORDUT_LOG_INFO("RKNN SDK version: {}, driver version: {}",
                     version.api_version, version.drv_version);
  rknn_input_output_num io_num;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  config_.input_tensors_count = io_num.n_input;
  config_.output_tensors_count = io_num.n_output;
  input_attr_ = new rknn_tensor_attr[io_num.n_input];
  output_attr_ = new rknn_tensor_attr[io_num.n_output];
  //  rknn_tensor_attr tensor_attr;
  input_ = new rknn_input[config_.input_tensors_count];
  memset(input_, 0, sizeof(rknn_input) * config_.input_tensors_count);
  for (int i = 0; i < io_num.n_input; ++i) {
    auto &tensor_attr = input_attr_[i];
    tensor_attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &tensor_attr,
                     sizeof(rknn_tensor_attr));
    input_[i].index = i;
    // 这里是RK的问题，不能使用模型读取的INT8，一定需要时UINT8
    input_[i].type = RKNN_TENSOR_UINT8;
    input_[i].fmt = tensor_attr.fmt;
    input_[i].size = tensor_attr.size;
    input_[i].buf = nullptr;
    config_.input_element_count[tensor_attr.name] = tensor_attr.n_elems;
    config_.scale[tensor_attr.name] = tensor_attr.scale;
    config_.zero_point[tensor_attr.name] = tensor_attr.zp;
    config_.tensor_size[tensor_attr.name] = tensor_attr.size_with_stride;
    if (tensor_attr.fmt == RKNN_TENSOR_NHWC) {
      auto width = tensor_attr.dims[2];
      auto stride = tensor_attr.w_stride;
      config_.width_equal_stride.emplace(tensor_attr.name, width == stride);
      config_.stride.emplace(tensor_attr.name, stride);
      KAYLORDUT_LOG_INFO("tensor name: {} width: {} stride: {}",
                         tensor_attr.name, width, stride);
    }
    KAYLORDUT_LOG_INFO("tensor name: {}, size with stride: {}",
                       tensor_attr.name, tensor_attr.size_with_stride);
    std::vector<int64_t> shape;
    for (int j = 0; j < tensor_attr.n_dims; ++j) {
      shape.push_back(tensor_attr.dims[j]);
    }
    config_.input_layer_shape.emplace(tensor_attr.name, shape);
    config_.input_single_element_size.emplace(
        tensor_attr.name, tensor_attr.size / tensor_attr.n_elems);
    config_.input_index_to_name.emplace(tensor_attr.index, tensor_attr.name);
  }
  output_ = new rknn_output[config_.output_tensors_count];
  memset(output_, 0, config_.output_tensors_count * sizeof(rknn_output));
  for (int i = 0; i < io_num.n_output; ++i) {
    auto &tensor_attr = output_attr_[i];
    tensor_attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &tensor_attr,
                     sizeof(rknn_tensor_attr));
    config_.output_element_count[tensor_attr.name] = tensor_attr.n_elems;
    config_.scale[tensor_attr.name] = tensor_attr.scale;
    config_.zero_point[tensor_attr.name] = tensor_attr.zp;
    config_.tensor_size[tensor_attr.name] = tensor_attr.size_with_stride;
    KAYLORDUT_LOG_INFO("tensor name: {}, size with stride: {}",
                       tensor_attr.name, tensor_attr.size_with_stride);
    std::vector<int64_t> shape;
    for (int j = 0; j < tensor_attr.n_dims; ++j) {
      shape.push_back(tensor_attr.dims[j]);
    }
    config_.output_layer_shape.emplace(tensor_attr.name, shape);
    config_.output_single_element_size.emplace(
        tensor_attr.name, tensor_attr.size / tensor_attr.n_elems);
    config_.output_index_to_name.emplace(tensor_attr.index, tensor_attr.name);
    auto is_quantized =
        (tensor_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
         RKNN_TENSOR_INT8)
            ? true
            : false;
    output_[i].want_float = (!is_quantized);
    output_[i].is_prealloc = true;
    output_[i].buf = nullptr;
    output_[i].size = tensor_attr.size;
  }
}

void Rk3588::BindInputAndOutput(ai_framework::TensorData &tensor_data) {
  if (zero_copy_) {
    auto input_data = tensor_data.get_input_rknn_tensor_mem_ptr();
    auto output_data = tensor_data.get_output_rknn_tensor_mem_ptr();
    int ret = 0;
    for (int i = 0; i < config_.input_tensors_count; ++i) {
      (input_attr_ + i)->type = RKNN_TENSOR_UINT8;
      ret = rknn_set_io_mem(ctx_, input_data[i], &input_attr_[i]);
      if (ret != RKNN_SUCC) {
        KAYLORDUT_LOG_ERROR("rknn_set_io_mem failed, error code = {}", ret);
        exit(EXIT_FAILURE);
      }
    }
    for (int i = 0; i < config_.output_tensors_count; ++i) {
      ret = rknn_set_io_mem(ctx_, output_data[i], &output_attr_[i]);
      if (ret != RKNN_SUCC) {
        KAYLORDUT_LOG_ERROR("rknn_set_io_mem failed, error code = {}", ret);
        exit(EXIT_FAILURE);
      }
    }
    KAYLORDUT_LOG_INFO("BindInputAndOutput rk3588 zero-copy mode successfully");
  } else {
    auto input_data = tensor_data.get_input_tensor_ptr();
    for (int i = 0; i < config_.input_tensors_count; ++i) {
      input_[i].buf = input_data[i];
    }
    auto output_data = tensor_data.get_output_tensor_ptr();
    for (int i = 0; i < config_.output_tensors_count; ++i) {
      output_[i].buf = output_data[i];
    }
  }
}

void Rk3588::DoInference() {
  int ret;
  if (!zero_copy_) {
    ret = rknn_inputs_set(ctx_, config_.input_tensors_count, input_);
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("rknn_input_set failed! error code = {}", ret);
      exit(EXIT_FAILURE);
    }
  }
  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    KAYLORDUT_LOG_ERROR("rknn_run failed, error code = {}", ret);
    exit(EXIT_FAILURE);
  }
  if (!zero_copy_) {
    ret =
        rknn_outputs_get(ctx_, config_.output_tensors_count, output_, nullptr);
    if (ret != RKNN_SUCC) {
      KAYLORDUT_LOG_ERROR("rknn_outputs_get failed, error code = {}", ret);
      exit(EXIT_FAILURE);
    }
  }
}
