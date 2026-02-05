#pragma once

#include "morozova_s_connected_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_connected_components {

class MorozovaSConnectedComponentsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MorozovaSConnectedComponentsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ProcessComponent(int start_i, int start_j, int current_label);

  int rows_{0};
  int cols_{0};
};

}  // namespace morozova_s_connected_components
