#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"
#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"
#include "morozova_s_connected_components/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozova_s_connected_components {

class MorozovaSRunFuncTestsConnectedComponents : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(test_params);
    std::string pattern = std::get<1>(test_params);
    if (size <= 0) {
      input_data_ = {};
      return;
    }
    input_data_ = std::vector<std::vector<int>>(size, std::vector<int>(size, 0));
    if (pattern == "cross") {
      for (int i = 0; i < size; ++i) {
        input_data_[i][size / 2] = 1;
        input_data_[size / 2][i] = 1;
      }
    } else if (pattern == "square") {
      for (int i = size / 4; i < 3 * size / 4; ++i) {
        for (int j = size / 4; j < 3 * size / 4; ++j) {
          input_data_[i][j] = 1;
        }
      }
    } else if (pattern == "dots") {
      for (int i = 1; i < size; i += 2) {
        for (int j = 1; j < size; j += 2) {
          input_data_[i][j] = 1;
        }
      }
    } else if (pattern == "border") {
      for (int i = 0; i < size; ++i) {
        input_data_[0][i] = 1;
        input_data_[size - 1][i] = 1;
        input_data_[i][0] = 1;
        input_data_[i][size - 1] = 1;
      }
    } else {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          if ((i + j) % 2 == 0) {
            input_data_[i][j] = 1;
          }
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }
    int reported_components = 0;
    if (!output_data.empty() && output_data.back().size() == 1) {
      reported_components = output_data.back()[0];
      output_data.pop_back();
    }
    if (output_data.size() != input_data_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < input_data_.size(); ++i) {
      if (output_data[i].size() != input_data_[i].size()) {
        return false;
      }
      for (std::size_t j = 0; j < input_data_[i].size(); ++j) {
        if (input_data_[i][j] == 1) {
          if (output_data[i][j] <= 0) {
            return false;
          }
        } else {
          if (output_data[i][j] != 0) {
            return false;
          }
        }
      }
    }

    std::vector<int> labels;
    for (const auto &row : output_data) {
      for (int label : row) {
        if (label > 0) {
          labels.push_back(label);
        }
      }
    }
    std::ranges::sort(labels);
    const auto [first, last] = std::ranges::unique(labels);
    labels.erase(first, last);

    if (!labels.empty()) {
      if (labels[0] != 1) {
        return false;
      }
      for (std::size_t i = 1; i < labels.size(); ++i) {
        if (labels[i] != labels[i - 1] + 1) {
          return false;
        }
      }
      if (labels.size() != static_cast<std::size_t>(reported_components)) {
        return false;
      }
    } else if (reported_components != 0) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MorozovaSRunFuncTestsConnectedComponents, ConnectedComponentsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(8, "cross"), std::make_tuple(10, "square"),
                                            std::make_tuple(12, "dots"), std::make_tuple(6, "border")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<MorozovaSConnectedComponentsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_morozova_s_connected_components),
                                           ppc::util::AddFuncTask<MorozovaSConnectedComponentsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_morozova_s_connected_components));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName =
    MorozovaSRunFuncTestsConnectedComponents::PrintFuncTestName<MorozovaSRunFuncTestsConnectedComponents>;

INSTANTIATE_TEST_SUITE_P(ConnectedComponentsTests, MorozovaSRunFuncTestsConnectedComponents, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace morozova_s_connected_components
