#pragma once

#include "morozova_s_connected_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_connected_components {

class MorozovaSConnectedComponentsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MorozovaSConnectedComponentsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BFSLabeling(int start_row, int start_col, int label);
  std::vector<std::pair<int, int>> GetNeighbors(int row, int col);
  void ExchangeBoundaryRows();
  void MergeLabels(int label1, int label2);

  std::vector<std::vector<int>> grid_;
  std::vector<std::vector<bool>> visited_;
  int rows_;
  int cols_;
};

}  // namespace morozova_s_connected_components
