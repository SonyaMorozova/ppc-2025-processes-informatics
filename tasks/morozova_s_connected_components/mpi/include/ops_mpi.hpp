#pragma once

#include <utility>
#include <vector>

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

  void LabelLocalComponents();
  void ProcessBoundaries();
  void MergeGlobalLabels();
  static std::pair<int, int> CalculateProcessBounds(int rows, int size, int process_rank);
  [[nodiscard]] std::vector<std::pair<int, int>> GetNeighbors(int row, int col) const;
  std::vector<std::vector<int>> grid_;
  std::vector<std::vector<bool>> visited_;
  int rows_{0};
  int cols_{0};
  int rank_{0};
  int size_{0};
  int start_row_{0};
  int end_row_{0};
};

}  // namespace morozova_s_connected_components
