//
// Created by ubuntu on 2/6/25.
//

#ifndef NNRT_H
#define NNRT_H
#include "acl/acl.h"
#include "ai_framework.h"

class Nnrt : public ai_framework::AiInstance {
 public:
  Nnrt() = default;
  ~Nnrt();
  virtual void Initialize(const char *model_path) final;
  virtual void BindInputAndOutput(ai_framework::TensorData &tensor_data) final;
  virtual void DoInference() final;
  void set_async(bool async) { async_ = async; }
  bool get_async() { return async_; }

 private:
  void InitResouce();
  bool async_ = false;
  ai_framework::TensorData *tensor_data_;
  aclmdlDataset *input_dataset_;
  std::vector<aclDataBuffer *> input_data_buffer_;
  aclmdlDataset *output_dataset_;
  std::vector<aclDataBuffer *> output_data_buffer_;
  uint32_t model_id_;
  int32_t device_id_{0};
  uint32_t count_{0};
  aclrtStream stream_;
  aclrtRunMode run_mode_;
  aclrtContext context_;
  aclmdlDesc *model_desc_{nullptr};
};

#endif  // NNRT_H
