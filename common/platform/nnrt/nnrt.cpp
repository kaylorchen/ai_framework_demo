//
// Created by ubuntu on 2/6/25.
//

#include "nnrt.h"

#include "kaylordut/log/logger.h"

void Nnrt::Initialize(const char *model_path) {
  InitResouce();
  config_.model_name_path = model_path;
  config_.model_format = ModelFormat::NNRT_FORMAT;
  auto ret = aclmdlLoadFromFile(model_path, &model_id_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Load model from file failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Load model from file success, model id: {}", model_id_);
  model_desc_ = aclmdlCreateDesc();
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Getting model desc failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Create acl model desc success");
  config_.input_tensors_count = aclmdlGetNumInputs(model_desc_);
  config_.output_tensors_count = aclmdlGetNumOutputs(model_desc_);
  for (int i = 0; i < config_.input_tensors_count; ++i) {
    aclmdlIODims dims;
    ret = aclmdlGetInputDimsV2(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      KAYLORDUT_LOG_ERROR("Getting input dims failed, ret = {}", ret);
    }
    KAYLORDUT_LOG_DEBUG("Getting input dims success");
    std::vector<int64_t> shape;
    size_t count = 1;
    for (int j = 0; j < dims.dimCount; ++j) {
      shape.push_back(dims.dims[j]);
      count *= dims.dims[j];
    }
    config_.input_layer_shape.emplace(dims.name, shape);
    config_.input_element_count.emplace(dims.name, count);
    auto size = aclmdlGetInputSizeByIndex(model_desc_, i);
    config_.tensor_size.emplace(dims.name, size);
    config_.input_single_element_size.emplace(dims.name, size / count);
    config_.input_index_to_name.emplace(i, dims.name);
  }
  for (int i = 0; i < config_.output_tensors_count; ++i) {
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      KAYLORDUT_LOG_ERROR("Getting output dims failed, ret = {}", ret);
    }
    KAYLORDUT_LOG_INFO("Getting output dims success, index = {}, name: {}", i,
                       dims.name);
    std::vector<int64_t> shape;
    size_t count = 1;
    for (int j = 0; j < dims.dimCount; ++j) {
      shape.push_back(dims.dims[j]);
      count *= dims.dims[j];
    }
    config_.output_layer_shape.emplace(dims.name, shape);
    config_.output_element_count.emplace(dims.name, count);
    auto size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    config_.tensor_size.emplace(dims.name, size);
    config_.output_single_element_size.emplace(dims.name, size / count);
    config_.output_index_to_name.emplace(i, dims.name);
  }
  this->PrintLayerInfo();
}

Nnrt::~Nnrt() {
  for (auto &kv : input_data_buffer_) {
    if (kv != nullptr) {
      aclDestroyDataBuffer(kv);
      kv = nullptr;
    }
  }
  for (auto &kv : output_data_buffer_) {
    if (kv != nullptr) {
      aclDestroyDataBuffer(kv);
      kv = nullptr;
    }
  }
  if (input_dataset_ != nullptr) {
    aclmdlDestroyDataset(input_dataset_);
    input_dataset_ = nullptr;
  }
  if (output_dataset_ != nullptr) {
    aclmdlDestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
  }
  if (model_desc_ != nullptr) {
    aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  if (model_id_ != 0) {
    aclmdlUnload(model_id_);
  }
  aclrtSynchronizeStream(stream_);
  aclrtDestroyStream(stream_);
  aclrtDestroyContext(context_);
  aclrtResetDevice(device_id_);
  aclFinalize();
}

void Nnrt::InitResouce() {
  aclError ret = ACL_SUCCESS;
  KAYLORDUT_TIME_COST_INFO("Acl Init", ret = aclInit(nullptr));
  if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
    KAYLORDUT_LOG_INFO("acl has initialized");
  } else if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Init acl failed, ret = {}", ret);
    exit(EXIT_FAILURE);
  }
  KAYLORDUT_LOG_INFO("Init acl success");
  int32_t version[3];
  ret = aclrtGetVersion(&version[0], &version[1], &version[2]);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Getting version failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Version: {}.{}.{}", version[0], version[1], version[2]);
  ret = aclrtGetDeviceCount(&count_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Get device count failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Get device count success, count: {}", count_);
  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Set device failed, ret = {}", ret);
    exit(EXIT_FAILURE);
  }
  KAYLORDUT_LOG_INFO("Set device success, device id = {}", device_id_);
  ret = aclrtGetRunMode(&run_mode_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Get run mode failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Get run mode success, mode: {}",
                     run_mode_ == aclrtRunMode::ACL_HOST ? "host" : "device");

  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Create context failed, ret = {}", ret);
  }
  KAYLORDUT_LOG_INFO("Create context success");
  // ret = aclrtCreateStream(&stream_);
  // if (ret != ACL_ERROR_NONE) {
  //   KAYLORDUT_LOG_ERROR("Create stream failed, ret = {}", ret);
  // }
  // KAYLORDUT_LOG_INFO("Create stream success");
}

void Nnrt::DoInference() {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_ERROR_NONE) {
    KAYLORDUT_LOG_ERROR("Set current context failed, ret = {}", ret);
  }
  if (run_mode_ == ACL_HOST) {
    auto input = this->tensor_data_->get_input_tensor_ptr();
    auto output = this->tensor_data_->get_output_tensor_ptr();
    auto input_device = this->tensor_data_->get_input_tensor_device_ptr();
    auto output_device = this->tensor_data_->get_output_tensor_device_ptr();
    auto input_size = this->tensor_data_->get_input_tensor_size();
    auto output_size = this->tensor_data_->get_output_tensor_size();
    for (int i = 0; i < this->get_input_tensor_count(); ++i) {
      if (run_mode_ == ACL_DEVICE) {
        auto ret = aclrtMemcpy(input_device[i], input_size.at(i), input[i],
                               input_size.at(i), ACL_MEMCPY_DEVICE_TO_DEVICE);
        while (ret != ACL_SUCCESS) {
          KAYLORDUT_LOG_ERROR("input acl memcpy failed, ret = {}", ret);
          ret = aclrtMemcpy(input_device[i], input_size.at(i), input[i],
                            input_size.at(i), ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
      }
    }
    ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
    if (ret != ACL_ERROR_NONE) {
      KAYLORDUT_LOG_ERROR("Execute model failed, ret = {}", ret);
    } else {
      KAYLORDUT_LOG_DEBUG("Execute model success");
      for (int i = 0; i < this->get_output_tensor_count(); ++i) {
        if (run_mode_ == ACL_DEVICE) {
          ret = aclrtMemcpy(output[i], output_size.at(i), output_device[i],
                            output_size.at(i), ACL_MEMCPY_DEVICE_TO_DEVICE);
          KAYLORDUT_LOG_ERROR_EXPRESSION(
              ret != ACL_SUCCESS, "output acl memcpy failed, ret ={}", ret);
        }
      }
    }
  } else {
    ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
    if (ret != ACL_SUCCESS) {
      KAYLORDUT_LOG_ERROR("Execute model failed, ret = {}", ret);
    }
  }
}

void Nnrt::BindInputAndOutput(ai_framework::TensorData &tensor_data) {
  this->tensor_data_ = &tensor_data;
  void **input;
  void **output;
  if (run_mode_ == ACL_HOST) {
    input = tensor_data.get_input_tensor_device_ptr();
    output = tensor_data.get_output_tensor_device_ptr();
  } else {
    input = tensor_data.get_input_tensor_ptr();
    output = tensor_data.get_output_tensor_ptr();
  }
  input_dataset_ = aclmdlCreateDataset();
  if (input_dataset_ == nullptr) {
    KAYLORDUT_LOG_ERROR("Create input dataset failed");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < config_.input_tensors_count; ++i) {
    auto size = config_.tensor_size.at(config_.input_index_to_name.at(i));
    auto *input_data = aclCreateDataBuffer(input[i], size);
    if (input_data == nullptr) {
      KAYLORDUT_LOG_ERROR("Create input data buffer failed, tensor name: {}",
                          config_.input_index_to_name.at(i));
      exit(EXIT_FAILURE);
    }
    input_data_buffer_.push_back(input_data);
    auto ret = aclmdlAddDatasetBuffer(input_dataset_, input_data);
    if (ret != ACL_ERROR_NONE) {
      KAYLORDUT_LOG_ERROR("Add dataset buffer failed, tensor name: {}",
                          config_.input_index_to_name.at(i));
      exit(EXIT_FAILURE);
    }
    KAYLORDUT_LOG_INFO("Add dataset buffer success, tensor name: {}",
                       config_.input_index_to_name.at(i));
  }
  output_dataset_ = aclmdlCreateDataset();
  for (int i = 0; i < config_.output_tensors_count; ++i) {
    auto size = config_.tensor_size.at(config_.output_index_to_name.at(i));
    auto *output_data = aclCreateDataBuffer(output[i], size);
    if (output_data == nullptr) {
      KAYLORDUT_LOG_ERROR("Create output data buffer failed, tensor name: {}",
                          config_.output_index_to_name.at(i));
      exit(EXIT_FAILURE);
    }
    output_data_buffer_.push_back(output_data);
    auto ret = aclmdlAddDatasetBuffer(output_dataset_, output_data);
    if (ret != ACL_ERROR_NONE) {
      KAYLORDUT_LOG_ERROR("Add dataset buffer failed, tensor name: {}",
                          config_.output_index_to_name.at(i));
      exit(EXIT_FAILURE);
    }
    KAYLORDUT_LOG_INFO("Add dataset buffer success, tensor name: {}",
                       config_.output_index_to_name.at(i));
  }
}
