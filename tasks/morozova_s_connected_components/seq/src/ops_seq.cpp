#include "morozova_s_connected_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stack>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {

MorozovaSConnectedComponentsSEQ::MorozovaSConnectedComponentsSEQ(const InType &in) : rows_(0), cols_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool MorozovaSConnectedComponentsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }
  std::size_t cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  rows_ = static_cast<int>(input.size());
  cols_ = static_cast<int>(input[0].size());
  grid_ = input;
  visited_.assign(rows_, std::vector<bool>(cols_, false));
  GetOutput() = std::vector<std::vector<int>>(rows_, std::vector<int>(cols_, 0));
  return true;
}

void MorozovaSConnectedComponentsSEQ::DFSLabeling(int row, int col, int label) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(row, col);
  visited_[row][col] = true;
  GetOutput()[row][col] = label;

  while (!stack.empty()) {
    auto [current_row, current_col] = stack.top();
    stack.pop();
    auto neighbors = GetNeighbors(current_row, current_col);

    for (const auto &neighbor : neighbors) {
      int nr = neighbor.first;
      int nc = neighbor.second;
      if (!visited_[nr][nc]) {
        visited_[nr][nc] = true;
        GetOutput()[nr][nc] = label;
        stack.emplace(nr, nc);
      }
    }
  }
}

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsSEQ::GetNeighbors(int row, int col) const {
  std::vector<std::pair<int, int>> neighbors;
  const std::array<int, 8> dr = {-1, -1, -1, 0, 0, 1, 1, 1};
  const std::array<int, 8> dc = {-1, 0, 1, -1, 1, -1, 0, 1};
  for (size_t i = 0; i < 8; ++i) {
    int new_row = row + dr[i];
    int new_col = col + dc[i];
    if (new_row >= 0 && new_row < rows_ && new_col >= 0 && new_col < cols_ && grid_[new_row][new_col] == 1) {
      neighbors.emplace_back(new_row, new_col);
    }
  }
  return neighbors;
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  int label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        DFSLabeling(i, j, label);
        ++label;
      }
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PostProcessingImpl() {
  int max_label = 0;
  for (const auto &row : GetOutput()) {
    for (int label : row) {
      max_label = std::max(label, max_label);
    }
  }
  auto &output = GetOutput();
  output.push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
