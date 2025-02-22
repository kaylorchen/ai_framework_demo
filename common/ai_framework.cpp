//
// Created by kaylor chen on 2024/6/9.
//
#include "ai_framework.h"
#include "kaylordut/log/logger.h"
#include "sstream"

size_t ai_framework::AiInstance::get_input_tensor_size() {
  size_t size = 0;
  for (const auto &kv : config_.input_element_count) {
    size += (kv.second * config_.input_single_element_size[kv.first]);
  }
  return size;
}

size_t ai_framework::AiInstance::get_output_tensor_size() {
  size_t size = 0;
  for (const auto &kv : config_.output_element_count) {
    size += (kv.second * config_.output_single_element_size[kv.first]);
  }
  return size;
}

uint16_t ai_framework::AiInstance::get_input_tensor_count() {
  return config_.input_tensors_count;
}

uint16_t ai_framework::AiInstance::get_output_tensor_count() {
  return config_.output_tensors_count;
}

void ai_framework::AiInstance::PrintLayerInfo() {
  KAYLORDUT_LOG_INFO("model path: {}", config_.model_name_path);
  {
    std::stringstream ss;
    ss << "Input layer information:\n";
    for (int count = 0; count < config_.input_tensors_count; ++count) {
      auto name = config_.input_index_to_name.at(count);
      ss << "name: " << name << " shape: {";
      for (int i = 0; i < config_.input_layer_shape.at(name).size(); ++i) {
        ss << config_.input_layer_shape.at(name).at(i);
        if (i != config_.input_layer_shape.at(name).size() - 1) {
          ss << ", ";
        }
      }
      ss << "} element_count: " << config_.input_element_count[name]
         << " total:"
         << config_.input_element_count[name] *
                config_.input_single_element_size[name]
         << " bytes ";
      ss << "zero_point: " << config_.zero_point[name]
         << " scale: " << config_.scale[name];
      ss << std::endl;
    }
    ss << "input layer size:" << get_input_tensor_size();
    ss << std::endl;
    KAYLORDUT_LOG_INFO("{}", ss.str());
  }
  {
    std::stringstream ss;
    ss << "Output layer information:\n";
    for (int count = 0; count < config_.output_tensors_count; ++count) {
      auto name = config_.output_index_to_name.at(count);
      ss << "name: " << name << " shape: {";
      for (int i = 0; i < config_.output_layer_shape.at(name).size(); ++i) {
        ss << config_.output_layer_shape.at(name).at(i);
        if (i != config_.output_layer_shape.at(name).size() - 1) {
          ss << ", ";
        }
      }
      ss << "} element_count: " << config_.output_element_count[name]
         << " total: "
         << config_.output_element_count[name] *
                config_.output_single_element_size[name]
         << " bytes ";
      ss << "zero_point: " << config_.zero_point[name]
         << " scale: " << config_.scale[name];
      ss << std::endl;
    }
    ss << "output layer size:" << get_output_tensor_size() << std::endl;
    KAYLORDUT_LOG_INFO("{}", ss.str());
  }
}
namespace ai_framework {
TensorData::TensorData(const ai_framework::Config &config) {
  KAYLORDUT_LOG_INFO("TensorData constructor is called");
  input_tensor_ptr_ = new void *[config.input_tensors_count];
#ifdef RK3588
  rknn_ctx_ = config.rknn_ctx;
  rknn_zero_copy_ = config.rknn_zero_copy;
  if (config.rknn_zero_copy) {
    input_rknn_tensor_mem_ptr_ =
        new rknn_tensor_mem *[config.input_tensors_count];
  }
#endif
#ifdef TRT
  input_tensor_cuda_ptr_ = new void *[config.input_tensors_count];
#endif
#ifdef NNRT
  if (!is_device_) {
    input_tensor_device_ptr_ = new void *[config.input_tensors_count];
  }
#endif
  for (int count = 0; count < config.input_tensors_count; ++count) {
    auto name = config.input_index_to_name.at(count);
    auto size = config.tensor_size.at(name);
    input_tensor_size_.push_back(size);
#ifdef RK3588
    if (config.rknn_zero_copy) {
      input_rknn_tensor_mem_ptr_[count] =
          rknn_create_mem(config.rknn_ctx, size);
      input_tensor_ptr_[count] =
          static_cast<uint8_t *>(input_rknn_tensor_mem_ptr_[count]->virt_addr);
      auto &ref = input_rknn_tensor_mem_ptr_[count];
      KAYLORDUT_LOG_INFO(
          "input tensor name: {}, virt_addr: 0x{:p}, phys_addr: "
          "0x{:x}, fd: {}, offset: {}, size: {}",
          name, ref->virt_addr, ref->phys_addr, ref->fd, ref->offset,
          ref->size);
    } else {
      input_tensor_ptr_[count] = new uint8_t[size];
    }
// #elif defined NNRT
//     if (config.is_device) {
//       is_device_ = config.is_device;
//       // auto ret = aclrtMalloc(&input_tensor_ptr_[count], size,
//       ACL_MEM_MALLOC_HUGE_FIRST); auto ret =
//       aclrtMallocHost(&input_tensor_ptr_[count], size); if (ret !=
//       ACL_ERROR_NONE) {
//         KAYLORDUT_LOG_ERROR("Failed to allocate memory for input tensor: {}",
//         name);
//       }
//       KAYLORDUT_LOG_INFO("Allocated memory for input tensor: {}, size: {}",
//       name, size);
//     }
#elif defined NNRT
    if (is_device_) {
      aclrtMallocHost(&input_tensor_ptr_[count], size);
    } else {
      input_tensor_ptr_[count] = new uint8_t[size];
    }
#else
    input_tensor_ptr_[count] = new uint8_t[size];
#endif

#ifdef TRT
    cudaMalloc(&input_tensor_cuda_ptr_[count], size);
#endif
#ifdef NNRT
    if (!is_device_) {
      aclrtMalloc(&input_tensor_device_ptr_[count], size,
                  ACL_MEM_MALLOC_HUGE_FIRST);
    }
#endif
    input_name_.push_back(name);
  }
  input_tensor_count_ = config.input_tensors_count;

  output_tensor_ptr_ = new void *[config.output_tensors_count];
#ifdef RK3588
  if (config.rknn_zero_copy) {
    output_rknn_tensor_mem_ptr_ =
        new rknn_tensor_mem *[config.output_tensors_count];
  }
#endif
#ifdef TRT
  output_tensor_cuda_ptr_ = new void *[config.output_tensors_count];
#endif
#ifdef NNRT
  if (!is_device_) {
    output_tensor_device_ptr_ = new void *[config.output_tensors_count];
  }
#endif
  for (int count = 0; count < config.output_tensors_count; ++count) {
    auto name = config.output_index_to_name.at(count);
    auto size = config.tensor_size.at(name);
    output_tensor_size_.push_back(size);
#ifdef RK3588
    if (config.rknn_zero_copy) {
      output_rknn_tensor_mem_ptr_[count] =
          rknn_create_mem(config.rknn_ctx, size);
      output_tensor_ptr_[count] =
          static_cast<uint8_t *>(output_rknn_tensor_mem_ptr_[count]->virt_addr);
      auto &ref = output_rknn_tensor_mem_ptr_[count];
      KAYLORDUT_LOG_INFO(
          "output tensor name: {}, virt_addr: 0x{:p}, phys_addr: "
          "0x{:x}, fd: {}, offset: {}, size: {}",
          name, ref->virt_addr, ref->phys_addr, ref->fd, ref->offset,
          ref->size);
    } else {
      output_tensor_ptr_[count] = new uint8_t[size];
    }
// #elif defined NNRT
//     if (config.is_device) {
//       // auto ret = aclrtMalloc(&output_tensor_ptr_[count], size,
//       ACL_MEM_MALLOC_HUGE_FIRST); auto ret =
//       aclrtMallocHost(&output_tensor_ptr_[count], size); if (ret !=
//       ACL_ERROR_NONE) {
//         KAYLORDUT_LOG_ERROR("Failed to allocate memory for output tensor:
//         {}", name);
//       }
//       KAYLORDUT_LOG_INFO("Allocated memory for output tensor: {}, size: {}",
//       name, size);
//     }
#elif defined NNRT
    if (is_device_) {
      aclrtMallocHost(&output_tensor_ptr_[count], size);
    } else {
      output_tensor_ptr_[count] = new uint8_t[size];
    }
#else
    output_tensor_ptr_[count] = new uint8_t[size];
#endif
#ifdef TRT
    cudaMalloc(&output_tensor_cuda_ptr_[count], size);
#endif
#ifdef NNRT
    if (!is_device_) {
      aclrtMalloc(&output_tensor_device_ptr_[count], size,
                  ACL_MEM_MALLOC_HUGE_FIRST);
    }
#endif
    output_name_.push_back(name);
  }
  output_tensor_count_ = config.output_tensors_count;
  KAYLORDUT_LOG_INFO("TensorData() returned");
}

TensorData::~TensorData() {
#ifdef RK3588
  if (rknn_zero_copy_) {
    for (int i = 0; i < input_tensor_count_; ++i) {
      if (input_tensor_ptr_[i] != nullptr) {
        input_tensor_ptr_[i] = nullptr;
        rknn_destroy_mem(rknn_ctx_, input_rknn_tensor_mem_ptr_[i]);
      }
    }
    if (input_rknn_tensor_mem_ptr_ != nullptr) {
      delete[] input_rknn_tensor_mem_ptr_;
    }
    for (int i = 0; i < output_tensor_count_; ++i) {
      if (output_tensor_ptr_[i] != nullptr) {
        output_tensor_ptr_[i] = nullptr;
        rknn_destroy_mem(rknn_ctx_, output_rknn_tensor_mem_ptr_[i]);
      }
    }
    if (output_rknn_tensor_mem_ptr_ != nullptr) {
      delete[] output_rknn_tensor_mem_ptr_;
    }
  }
#endif
#ifdef NNRT
  if (!is_device_) {
    // 如果是HOST的话，device_ptr需要申请内存，所以这里释放
    for (int i = 0; i < input_tensor_count_; ++i) {
      if (input_tensor_device_ptr_ != nullptr) {
        if (input_tensor_device_ptr_[i] != nullptr) {
          aclrtFree(input_tensor_device_ptr_[i]);
          input_tensor_device_ptr_[i] = nullptr;
        }
      }
    }
    delete[] input_tensor_device_ptr_;
    for (int i = 0; i < output_tensor_count_; ++i) {
      if (output_tensor_device_ptr_[i] != nullptr) {
        aclrtFree(output_tensor_device_ptr_[i]);
        output_tensor_device_ptr_[i] = nullptr;
      }
    }
    delete[] output_tensor_device_ptr_;
  }
  // 如果仅仅是在device上运行，我们需要用aclrtFreeHost释放内存，这里是RC模式，所以可以减少内存拷贝
  for (int i = 0; i < input_tensor_count_; ++i) {
    if (input_tensor_ptr_[i] != nullptr && is_device_) {
      aclrtFreeHost(input_tensor_ptr_[i]);
      input_tensor_ptr_[i] = nullptr;
    }
  }
  for (int i = 0; i < output_tensor_count_; ++i) {
    if (output_tensor_ptr_[i] != nullptr && is_device_) {
      aclrtFreeHost(output_tensor_ptr_[i]);
      output_tensor_ptr_[i] = nullptr;
    }
  }
#endif
  for (int i = 0; i < input_tensor_count_; ++i) {
    if (input_tensor_ptr_[i] != nullptr) {
      delete[] input_tensor_ptr_[i];
    }
  }
  if (input_tensor_ptr_ != nullptr) {
    delete[] input_tensor_ptr_;
  }
  for (int i = 0; i < output_tensor_count_; ++i) {
    if (output_tensor_ptr_[i] != nullptr) {
      delete[] output_tensor_ptr_[i];
    }
  }
  if (output_tensor_ptr_ != nullptr) {
    delete[] output_tensor_ptr_;
  }
#ifdef TRT
  for (int i = 0; i < input_tensor_count_; ++i) {
    cudaFree(input_tensor_cuda_ptr_[i]);
  }
  delete[] input_tensor_cuda_ptr_;
  for (int i = 0; i < output_tensor_count_; ++i) {
    cudaFree(output_tensor_cuda_ptr_[i]);
  }
  delete[] output_tensor_cuda_ptr_;
#endif
}
} // namespace ai_framework
