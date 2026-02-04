#pragma once

#include <unordered_map>
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

  void FloodFill(int row, int col, int label);
  [[nodiscard]] std::vector<std::pair<int, int>> GetNeighbors(int row, int col) const;

  void InitMPI();
  [[nodiscard]] std::pair<int, int> ComputeRowRange() const;
  void ComputeLocalComponents(int start_row, int end_row, int base_label);
  void GatherLocalResults();
  void MergeBoundaries();
  void ProcessBoundaryCell(int proc, int j, int dj, std::unordered_map<int, int> &parent);
  static int FindRoot(std::unordered_map<int, int> &parent, int v);
  void NormalizeLabels();
  void BroadcastResult();
  void SendLocalResult(int start_row, int end_row);
  void ReceiveFinalResult();

  int rank_{0};
  int size_{1};
  int rows_{0};
  int cols_{0};
  int rows_per_proc_{0};
  int remainder_{0};
};

}  // namespace morozova_s_connected_components
