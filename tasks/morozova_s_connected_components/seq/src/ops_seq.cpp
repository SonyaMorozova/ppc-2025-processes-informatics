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
  GetOutput().assign(GetInput().size(), std::vector<int>(GetInput().front().size(), 0));
  return true;
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int rows = static_cast<int>(input.size());
  const int cols = static_cast<int>(input.front().size());

  int current_label = 1;
  std::queue<std::pair<int, int>> q;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (input[i][j] != 0 && output[i][j] == 0) {
        output[i][j] = current_label;
        q.push({i, j});

        while (!q.empty()) {
          const auto [x, y] = q.front();
          q.pop();

          for (const auto &[dx, dy] : kShifts) {
            const int nx = x + dx;
            const int ny = y + dy;

            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && input[nx][ny] != 0 && output[nx][ny] == 0) {
              output[nx][ny] = current_label;
              q.push({nx, ny});
            }
          }
        }
        ++current_label;
      }
    }
  }

  return true;
}

bool MorozovaSConnectedComponentsSEQ::PostProcessingImpl() {
  int max_label = 0;
  for (const auto &row : GetOutput()) {
    for (const int v : row) {
      max_label = std::max(max_label, v);
    }
  }
  return max_label > 0;
}

}  // namespace morozova_s_connected_components
