#include "morozova_s_connected_components/seq/include/ops_seq.hpp"

#include <array>
#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {
namespace {

std::vector<std::pair<int, int>> GetNeighborsSeq(int r, int c, int rows, int cols) {
  constexpr std::array<std::pair<int, int>, 8> shifts = {
      {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};
  std::vector<std::pair<int, int>> result;
  result.reserve(8);
  for (const auto &sh : shifts) {
    const int nr = r + sh.first;
    const int nc = c + sh.second;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
      result.emplace_back(nr, nc);
    }
  }

  return result;
}

}  // namespace

MorozovaSConnectedComponentsSEQ::MorozovaSConnectedComponentsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool MorozovaSConnectedComponentsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }
  const std::size_t cols = input.front().size();
  return std::all_of(input.begin(), input.end(), [cols](const std::vector<int> &row) { return row.size() == cols; });
}
bool MorozovaSConnectedComponentsSEQ::PreProcessingImpl() {
  grid_ = GetInput();
  rows_ = static_cast<int>(grid_.size());
  cols_ = static_cast<int>(grid_.front().size());
  visited_.assign(rows_, std::vector<bool>(cols_, false));
  GetOutput().assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

void MorozovaSConnectedComponentsSEQ::DFSLabeling(int row, int col, int label) {
  std::queue<std::pair<int, int>> q;
  q.emplace(row, col);
  visited_[row][col] = true;
  GetOutput()[row][col] = label;
  while (!q.empty()) {
    const auto [r, c] = q.front();
    q.pop();
    for (const auto &[nr, nc] : GetNeighborsSeq(r, c, rows_, cols_)) {
      if (!visited_[nr][nc] && grid_[nr][nc] == 1) {
        visited_[nr][nc] = true;
        GetOutput()[nr][nc] = label;
        q.emplace(nr, nc);
      }
    }
  }
}

void MorozovaSConnectedComponentsSEQ::LabelComponents() {
  int label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        DFSLabeling(i, j, label);
        ++label;
      }
    }
  }
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  LabelComponents();
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_connected_components
