#include "morozova_s_connected_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {

namespace {

constexpr std::array<std::pair<int, int>, 8> kShifts = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};

}  // namespace

bool MorozovaSConnectedComponentsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }

  const std::size_t cols = input.front().size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
    for (int v : row) {
      if (v != 0 && v != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  rows_ = static_cast<int>(input.size());
  if (rows_ == 0) {
    cols_ = 0;
    GetOutput().clear();
    return true;
  }

  cols_ = static_cast<int>(input.front().size());
  grid_ = input;
  visited_.assign(rows_, std::vector<bool>(cols_, false));
  GetOutput().assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

  int current_label = 0;

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        ++current_label;
        std::queue<std::pair<int, int>> q;
        q.emplace(i, j);
        visited_[i][j] = true;
        GetOutput()[i][j] = current_label;

        while (!q.empty()) {
          const auto [r, c] = q.front();
          q.pop();

          for (const auto &[dr, dc] : kShifts) {
            const int nr = r + dr;
            const int nc = c + dc;

            if (nr < 0 || nr >= rows_ || nc < 0 || nc >= cols_) {
              continue;
            }

            if (grid_[nr][nc] == 1 && !visited_[nr][nc]) {
              visited_[nr][nc] = true;
              GetOutput()[nr][nc] = current_label;
              q.emplace(nr, nc);
            }
          }
        }
      }
    }
  }

  return true;
}

bool MorozovaSConnectedComponentsSEQ::PostProcessingImpl() {
  int max_label = 0;
  for (const auto &row : GetOutput()) {
    for (int v : row) {
      max_label = std::max(max_label, v);
    }
  }

  GetOutput().push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
