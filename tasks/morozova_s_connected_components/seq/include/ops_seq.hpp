#pragma once

#include <utility>
#include <vector>

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
  void DFSLabeling(int row, int col, int label);
  std::vector<std::pair<int, int>> GetNeighbors(int row, int col) const;
  std::vector<std::vector<int>> grid_;
  std::vector<std::vector<bool>> visited_;
  int rows_;
  int cols_;
};
}  // namespace morozova_s_connected_components
