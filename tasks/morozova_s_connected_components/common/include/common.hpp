#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace morozova_s_connected_components {
using InType = std::vector<std::vector<int>>;
using OutType = std::vector<std::vector<int>>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;
}  // namespace morozova_s_connected_components
