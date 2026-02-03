#include "morozova_s_connected_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {
MorozovaSConnectedComponentsSEQ::MorozovaSConnectedComponentsSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

namespace {
constexpr std::array<std::pair<int, int>, 8> kShifts = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};

}  // namespace

bool MorozovaSConnectedComponentsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }
  const std::size_t cols = input.front().size();
  if (cols == 0) {
    return false;
  }
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  rows_ = input.size();
  cols_ = input.front().size();
  labels_.assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  const auto &input = GetInput();
  int current_label = 0;

  for (std::size_t i = 0; i < rows_; ++i) {
    for (std::size_t j = 0; j < cols_; ++j) {
      if (input[i][j] != 0 && labels_[i][j] == 0) {
        ++current_label;
        std::queue<std::pair<std::size_t, std::size_t>> q;
        q.emplace(i, j);
        labels_[i][j] = current_label;

        while (!q.empty()) {
          const auto [x, y] = q.front();
          q.pop();

          for (const auto &[dx, dy] : kShifts) {
            const int nx = static_cast<int>(x) + dx;
            const int ny = static_cast<int>(y) + dy;

            if (nx >= 0 && ny >= 0 && nx < static_cast<int>(rows_) && ny < static_cast<int>(cols_)) {
              const auto ux = static_cast<std::size_t>(nx);
              const auto uy = static_cast<std::size_t>(ny);
              if (input[ux][uy] != 0 && labels_[ux][uy] == 0) {
                labels_[ux][uy] = current_label;
                q.emplace(ux, uy);
              }
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
  for (const auto &row : labels_) {
    for (const int v : row) {
      max_label = std::max(max_label, v);
    }
  }
  GetOutput() = max_label;
  return true;
}

}  // namespace morozova_s_connected_components
